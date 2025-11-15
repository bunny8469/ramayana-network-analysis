#!/usr/bin/env python3
import json
import re
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain # FIX for: AttributeError: module 'community' has no attribute 'best_partition'

# --- Configuration ---
FILE_TEXT = './../data/ramayana_resolved_final.json' # Using relative path for portability
FILE_CANONICAL_NAMES = './../data/canonical_characters.json' # Using relative path for portability
CO_OCCURRENCE_WINDOW = 15  # Max words distance for an undirected edge
OUTPUT_GRAPH_DATA = 'ramayana_graphs_and_centrality.json'

# --- Visualization Configuration (CLUSTER HEROES) ---
# Filter the graph to show only the most central nodes
TOP_N_NODES_TO_PLOT = 150 
# Number of nodes to label for clarity
TOP_N_LABELS = 25 
# Node size scaling (Min/Max size in the plot for NON-HEROES)
MIN_NODE_SIZE = 400  # Increased for better visibility
MAX_NODE_SIZE = 10000 # Increased for better contrast 
# Fixed size/font for the single most central character in each of the top N clusters
NUM_TOP_CLUSTERS_TO_HIGHLIGHT = 10
HERO_NODE_SIZE = 15000 # Massive fixed size for primary characters
HERO_FONT_SIZE = 20    # Largest font size


# Discourse markers for inferring speech direction (Source -> Target)
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)


# --- 1. Data Loading and Preparation ---

def load_data():
    """Loads all necessary JSON files."""
    try:
        # NOTE: Using simplified relative paths for file access
        with open(FILE_TEXT, 'r') as f:
            text_data = json.load(f)
        with open(FILE_CANONICAL_NAMES, 'r') as f:
            canonical_names = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required file not found. Please ensure '{e.filename}' is in the current directory.")
        raise
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from one of the input files.")
        raise

    # Convert the list of canonical names into a set for O(1) lookups
    character_set = set(canonical_names)
    return text_data, character_set

def tokenize_and_tag(canto_text, character_set):
    """Tokenizes text, replaces newlines/excess spaces, and tags words as characters/non-characters."""
    words = re.findall(r'\b\w+\b', canto_text.strip())
    tagged_tokens = []
    for word in words:
        is_char = word in character_set
        tagged_tokens.append((word, is_char))
    
    return tagged_tokens

# --- 2. Graph Construction (Co-occurrence and Conversational) ---

def build_undirected_graph(text_data, character_set):
    """Builds the Undirected (Co-occurrence) Graph."""
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(character_set)
    co_occurrence_edges = Counter()

    print("Building Undirected Graph (Co-occurrence)...")
    for chapter in text_data['chapters']:
        for canto_text in chapter['cantos']:
            tagged_tokens = tokenize_and_tag(canto_text, character_set)
            
            char_positions = [(i, token[0]) for i, token in enumerate(tagged_tokens) if token[1]]
            
            for i in range(len(char_positions)):
                char_A = char_positions[i][1]
                pos_A = char_positions[i][0]
                
                for j in range(i + 1, len(char_positions)):
                    char_B = char_positions[j][1]
                    pos_B = char_positions[j][0]

                    if pos_B - pos_A > CO_OCCURRENCE_WINDOW:
                        break 
                    
                    if char_A != char_B:
                        edge = tuple(sorted((char_A, char_B)))
                        co_occurrence_edges[edge] += 1

    for (u, v), weight in co_occurrence_edges.items():
        G_undirected.add_edge(u, v, weight=weight)
        
    return G_undirected, co_occurrence_edges

def build_directed_graph(text_data, character_set):
    """Builds the Directed (Conversational) Graph using discourse markers."""
    G_directed = nx.DiGraph()
    G_directed.add_nodes_from(character_set)
    conversational_edges = Counter()

    print("Building Directed Graph (Conversational)...")
    for chapter in text_data['chapters']:
        for canto_text in chapter['cantos']:
            tagged_tokens = tokenize_and_tag(canto_text, character_set)
            
            for i, (word, is_char) in enumerate(tagged_tokens):
                if is_char:
                    char_A = word # Potential speaker (Source)
                    
                    for j in range(i + 1, min(i + CO_OCCURRENCE_WINDOW, len(tagged_tokens))):
                        if DISCOURSE_MARKER_PATTERN.search(tagged_tokens[j][0]):
                            
                            for k in range(j + 1, min(j + 5, len(tagged_tokens))):
                                if tagged_tokens[k][1] and tagged_tokens[k][0] != char_A:
                                    char_B = tagged_tokens[k][0] # Conversational Target
                                    
                                    conversational_edges[(char_A, char_B)] += 1
                                    break 

    for (u, v), weight in conversational_edges.items():
        G_directed.add_edge(u, v, weight=weight)
        
    return G_directed, conversational_edges

# --- 3. Centrality Calculation ---

def calculate_centrality(G, graph_type):
    """Calculates Degree and Eigenvector centrality for a graph."""
    print(f"Calculating Centrality for {graph_type} Graph...")
    
    if graph_type == 'Directed':
        degree_centrality = nx.in_degree_centrality(G)
    else:
        degree_centrality = nx.degree_centrality(G)

    try:
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.NetworkXNoConvergence:
        print(f"Warning: Eigenvector centrality failed to converge for {graph_type}. Using unweighted calculation.")
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception as e:
        print(f"Error during Eigenvector centrality calculation for {graph_type}: {e}")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}

    centrality_results = {}
    for node in G.nodes():
        centrality_results[node] = {
            'degree_centrality': degree_centrality.get(node, 0.0),
            'eigenvector_centrality': eigenvector_centrality.get(node, 0.0),
        }
        
    return centrality_results

# --- 4. Community Detection and Visualization ---

def detect_communities(G):
    """Detects communities using the Louvain algorithm."""
    print("Detecting communities (Louvain algorithm)...")
    partition = community_louvain.best_partition(G, weight='weight', randomize=False)
    modularity = community_louvain.modularity(partition, G, weight='weight')
    
    community_map = {node: partition[node] for node in G.nodes()}
    
    return community_map, modularity

def identify_cluster_heroes(G_centrality_data, community_map, num_clusters=NUM_TOP_CLUSTERS_TO_HIGHLIGHT):
    """Identifies the most central character (by Eigenvector Centrality) for the top N largest clusters."""
    # Group nodes by community
    community_groups = Counter(community_map.values())
    
    # Get top N largest community IDs
    top_community_ids = [cid for cid, count in community_groups.most_common(num_clusters)]
    
    # Map community ID to its members
    id_to_members = {i: [node for node, cid in community_map.items() if cid == i] for i in top_community_ids}
    
    cluster_heroes = {} # Format: {hero_name: community_id}
    for cid, members in id_to_members.items():
        if members:
            # Find the node with the max Eigenvector Centrality in this community
            best_hero = max(members, key=lambda node: G_centrality_data.get(node, {}).get('eigenvector_centrality', 0))
            cluster_heroes[best_hero] = cid
            
    return cluster_heroes


def draw_graph(G, centrality_scores, community_map, cluster_heroes, title, filename, is_directed=False):
    """Generates and saves the filtered and styled network visualization."""
    print(f"Generating plot: {title}...")
    
    # 1. Filter Graph for Visualization (Top N Nodes)
    
    # Sort nodes by Eigenvector Centrality
    sorted_nodes = sorted(centrality_scores.items(), key=lambda item: item[1], reverse=True)
    top_nodes_names = [node for node, score in sorted_nodes[:TOP_N_NODES_TO_PLOT]]
    
    # Create a subgraph containing only the top nodes and their edges
    G_vis = G.subgraph(top_nodes_names).copy()
    
    # Filter centrality and community maps to match the subgraph
    centrality_vis = {n: centrality_scores[n] for n in G_vis.nodes()}
    community_vis = {n: community_map[n] for n in G_vis.nodes()}
    
    if not G_vis.nodes():
        print(f"Warning: Visualization skipped for {filename}. No nodes left after filtering.")
        return
        
    # 2. Prepare Node Sizes and Colors
    
    scores = list(centrality_vis.values())
    min_score = min(scores) if scores else 0
    max_score = max(scores) if scores else 1
    
    node_sizes = []
    
    for node in G_vis.nodes():
        if node in cluster_heroes:
            size = HERO_NODE_SIZE # Tier 1: Fixed, massive size for the hero
        else:
            score = centrality_vis.get(node, 0)
            
            # Scale score aggressively using the new constants
            if max_score > min_score:
                size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * (score - min_score) / (max_score - min_score)
            else:
                size = MIN_NODE_SIZE
        node_sizes.append(size)

    # 3. Prepare Colors based on Communities
    num_communities = max(community_vis.values()) + 1
    # Note: Suppressing MatplotlibDeprecationWarning for get_cmap
    with plt.rc_context({'figure.max_open_warning': 0}):
        try:
            color_map = plt.cm.get_cmap('Spectral', num_communities)
        except AttributeError:
             # Fallback for newer matplotlib versions (e.g., 3.11+)
            color_map = plt.colormaps.get_cmap('Spectral', num_communities)
    node_colors = [color_map(community_vis[node]) for node in G_vis.nodes()]

    # 4. Plotting
    plt.figure(figsize=(25, 20)) # Increased figure size
    plt.title(title, fontsize=24)
    
    # Use Fruchterman-Reingold layout for better node separation
    # k=0.08 (highly repulsive) and iterations=100 (high convergence) should maximize cluster separation.
    pos = nx.spring_layout(G_vis, k=0.08, iterations=100, seed=42) 
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G_vis, pos, 
        node_size=node_sizes, 
        node_color=node_colors, 
        alpha=0.8, 
        linewidths=1.5, 
        edgecolors='black'
    )
    
    # Draw edges
    # Calculate max edge weight for scaling
    max_weight = max([G_vis.get_edge_data(u, v).get('weight', 1) for u, v in G_vis.edges()]) if G_vis.edges() else 1
    # Increased base edge width slightly
    edge_widths = [(G_vis.get_edge_data(u, v).get('weight', 1) / max_weight) * 4 + 0.5 for u, v in G_vis.edges()]

    if is_directed:
        nx.draw_networkx_edges(
            G_vis, pos, 
            width=edge_widths,
            alpha=0.4,
            edge_color='darkgray',
            arrows=True,
            arrowsize=15 # Increased arrow size
        )
    else:
        nx.draw_networkx_edges(
            G_vis, pos, 
            width=edge_widths,
            alpha=0.4,
            edge_color='darkgray'
        )
        
    # 5. Draw Labels
    
    # Priority 1: All cluster heroes must be labeled, using the largest font.
    hero_labels = {hero: hero for hero in cluster_heroes if hero in G_vis.nodes()}
    
    # Priority 2: Fill the rest of the TOP_N_LABELS quota with the next most central nodes (excluding heroes).
    centrality_sorted = sorted(centrality_vis.items(), key=lambda item: item[1], reverse=True)
    central_nodes_not_heroes = [
        node for node, score in centrality_sorted
        if node not in cluster_heroes and node in G_vis.nodes()
    ]
    
    remaining_quota = TOP_N_LABELS - len(hero_labels)
    normal_labels = {node: node for node in central_nodes_not_heroes[:remaining_quota]}
    
    # Draw normal labels (standard font)
    nx.draw_networkx_labels(G_vis, pos, normal_labels, font_size=17, font_weight='bold', font_color='black') # Increased font size
    
    # Draw hero labels (largest font, highlighted color)
    nx.draw_networkx_labels(G_vis, pos, hero_labels, font_size=HERO_FONT_SIZE, font_weight='extra bold', font_color='black', bbox=dict(facecolor='yellow', alpha=0.6, boxstyle="round,pad=0.2"))
    
    # Add a simple legend for communities
    
    # --- Modularity FIX ---
    G_modularity = G_vis
    if is_directed:
        # Convert to undirected graph before calculating modularity, which requires an undirected type.
        G_modularity = G_vis.to_undirected(reciprocal=False)
    
    G_vis.modularity = community_louvain.modularity(community_vis, G_modularity, weight='weight')
    # --- END FIX ---
    
    for comm_id in range(num_communities):
        # We only want to label the communities that are actually represented in the plot (top 150 nodes)
        if comm_id in community_vis.values():
            plt.scatter([], [], color=color_map(comm_id), label=f'Community {comm_id + 1}')
    
    # Set the location of the legend outside the graph
    plt.legend(scatterpoints=1, frameon=True, labelspacing=1, title=f'Communities (Modularity: {G_vis.modularity:.2f})', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300) # Increased DPI for better image quality
    plt.close()
    print(f"Plot saved as {filename}")


# --- Main Execution ---

def main():
    try:
        text_data, character_set = load_data()
    except Exception:
        return

    # Filter out characters that never appear in the text
    all_appearing_chars = set()
    for chapter in text_data['chapters']:
        for canto_text in chapter['cantos']:
            for word in re.findall(r'\b\w+\b', canto_text):
                if word in character_set:
                    all_appearing_chars.add(word)
    
    character_set_filtered = all_appearing_chars

    # 1. Build Graphs
    G_undirected, co_occurrence_edges = build_undirected_graph(text_data, character_set_filtered)
    G_directed, conversational_edges = build_directed_graph(text_data, character_set_filtered)
    
    # Filter out isolated nodes
    G_undirected_filtered = G_undirected.subgraph([n for n, d in G_undirected.degree() if d > 0]).copy()
    G_directed_filtered = G_directed.subgraph([n for n in G_directed.nodes() if G_directed.in_degree(n) > 0 or G_directed.out_degree(n) > 0]).copy()

    # 2. Centrality Measures
    undirected_centrality = calculate_centrality(G_undirected_filtered, 'Undirected')
    directed_centrality = calculate_centrality(G_directed_filtered, 'Directed')
    
    # 3. Community Detection
    undirected_community_map, undirected_modularity = detect_communities(G_undirected_filtered)
    
    G_directed_unweighted = G_directed_filtered.to_undirected(reciprocal=False)
    G_directed_unweighted = nx.Graph(G_directed_unweighted) 
    directed_community_map, directed_modularity = detect_communities(G_directed_unweighted)

    # Extract Eigenvector scores for hero identification
    undirected_centrality_flat = {node: data['eigenvector_centrality'] for node, data in undirected_centrality.items()}
    directed_centrality_flat = {node: data['eigenvector_centrality'] for node, data in directed_centrality.items()}
    
    # 3.5 Identify Cluster Heroes
    undirected_heroes = identify_cluster_heroes(undirected_centrality, undirected_community_map, NUM_TOP_CLUSTERS_TO_HIGHLIGHT)
    directed_heroes = identify_cluster_heroes(directed_centrality, directed_community_map, NUM_TOP_CLUSTERS_TO_HIGHLIGHT)


    # 4. Visualization
    draw_graph(
        G_undirected_filtered, 
        undirected_centrality_flat, 
        undirected_community_map, 
        undirected_heroes, # PASSING HEROES
        f'Ramayana Undirected Co-occurrence Network (Top {TOP_N_NODES_TO_PLOT} Nodes)',
        'ramayana_undirected_network_filtered.png',
        is_directed=False
    )
    
    draw_graph(
        G_directed_filtered, 
        directed_centrality_flat, 
        directed_community_map, 
        directed_heroes, # PASSING HEROES
        f'Ramayana Directed Conversation Network (Top {TOP_N_NODES_TO_PLOT} Nodes)',
        'ramayana_directed_network_filtered.png',
        is_directed=True
    )
    
    # 5. Prepare final output structure
    results = {
        "metadata": {
            "total_canonical_characters": len(character_set),
            "total_appearing_characters": len(character_set_filtered),
            "co_occurrence_window": CO_OCCURRENCE_WINDOW
        },
        "undirected_network": {
            "nodes": len(G_undirected_filtered.nodes()),
            "edges": len(G_undirected_filtered.edges()),
            "modularity": undirected_modularity,
            "communities": Counter(undirected_community_map.values()),
            "top_10_centrality": sorted(undirected_centrality.items(), key=lambda item: item[1]['eigenvector_centrality'], reverse=True)[:10]
        },
        "directed_network": {
            "nodes": len(G_directed_filtered.nodes()),
            "edges": len(G_directed_filtered.edges()),
            "modularity": directed_modularity,
            "communities": Counter(directed_community_map.values()),
            "top_10_centrality": sorted(directed_centrality.items(), key=lambda item: item[1]['eigenvector_centrality'], reverse=True)[:10]
        },
        "centrality_results_undirected": undirected_centrality,
        "centrality_results_directed": directed_centrality,
        "undirected_edges_top_10": sorted(co_occurrence_edges.items(), key=lambda item: item[1], reverse=True)[:10],
        "directed_edges_top_10": sorted(conversational_edges.items(), key=lambda item: item[1], reverse=True)[:10]
    }

    # Save the final analysis to a JSON file
    with open(OUTPUT_GRAPH_DATA, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n✅ Network analysis and visualization data complete. Results saved to: {OUTPUT_GRAPH_DATA}")
    print("\n--- Summary ---")
    print(f"Undirected Graph: Nodes={results['undirected_network']['nodes']}, Edges={results['undirected_network']['edges']}, Communities={len(results['undirected_network']['communities'])}")
    print(f"Directed Graph: Nodes={results['directed_network']['nodes']}, Edges={results['directed_network']['edges']}, Communities={len(results['directed_network']['communities'])}")


if __name__ == '__main__':
    main()