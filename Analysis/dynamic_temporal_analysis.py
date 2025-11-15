import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os # For file handling
import imageio.v2 as imageio # For GIF generation
import pandas as pd # For CSV export

# --- Configuration ---
FILE_TEXT = './../data/ramayana_resolved_final.json'
FILE_CANONICAL_NAMES = './../data/canonical_characters.json'
CO_OCCURRENCE_WINDOW = 15 # Max words distance for an undirected edge
OUTPUT_ANALYSIS_DATA = 'ramayana_temporal_analysis.json'
# Fixed list of protagonists is now largely superseded by fractal analysis, but kept for legacy tracking
PROTAGONISTS_TO_TRACK = ["Rama", "Sita", "Hanuman", "Lakshmana", "Ravana", "Kausalya", "Dasharatha"] 
TOP_N_FRACTAL_PROTAGONISTS = 30 # Number of fractal protagonists to track and display in GIF/Summary
LINE_PLOT_N_PROTAGONISTS = 6 # Number of fractal protagonists to display in the line plot (for clarity)

# --- GIF Configuration ---
GIF_FRAME_DURATION = 2000 # Increased duration for better viewing
GIF_FILENAME_UNDIRECTED = 'fractal_network_undirected_evolution.gif'
GIF_FILENAME_DIRECTED = 'fractal_network_directed_evolution.gif'
TEMP_FRAME_DIR = 'temp_frames/'

# --- CSV Configuration ---
CSV_DIR = 'kanda_networks_csv/'

# Discourse markers for conversational graph (Source -> Target)
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- Utility Functions (unchanged) ---

def load_data():
    """Loads text and canonical names."""
    try:
        with open(FILE_TEXT, 'r') as f:
            text_data = json.load(f)
        with open(FILE_CANONICAL_NAMES, 'r') as f:
            canonical_names = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Required file not found: {e.filename}")
        raise
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from one of the input files.")
        raise
    
    character_set = set(canonical_names)
    return text_data, character_set

def tokenize_and_tag(canto_text, character_set):
    """Tokenizes text and tags words as characters."""
    words = re.findall(r'\b\w+\b', canto_text.strip())
    tagged_tokens = []
    for word in words:
        is_char = word in character_set
        tagged_tokens.append((word, is_char))
    return tagged_tokens

# --- Graph Building Functions (unchanged) ---

def build_graph_for_kanda(kanda_data, character_set, graph_type='undirected'):
    """Builds a single graph (undirected or directed) based on a Kanda's text."""
    
    G = nx.Graph() if graph_type == 'undirected' else nx.DiGraph()
    G.add_nodes_from(character_set)
    edges = Counter()
    
    # 1. Collect all canto texts in this Kanda
    canto_texts = kanda_data.get('cantos', [])
    
    # 2. Process each canto
    for canto_text in canto_texts:
        tagged_tokens = tokenize_and_tag(canto_text, character_set)
        
        # --- Undirected (Co-occurrence) Logic ---
        if graph_type == 'undirected':
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
                        edges[edge] += 1
                        
        # --- Directed (Conversational) Logic ---
        elif graph_type == 'directed':
             for i, (word, is_char) in enumerate(tagged_tokens):
                if is_char:
                    char_A = word # Potential speaker (Source)
                    
                    for j in range(i + 1, min(i + CO_OCCURRENCE_WINDOW, len(tagged_tokens))):
                        if DISCOURSE_MARKER_PATTERN.search(tagged_tokens[j][0]):
                            
                            for k in range(j + 1, min(j + 5, len(tagged_tokens))):
                                if tagged_tokens[k][1] and tagged_tokens[k][0] != char_A:
                                    char_B = tagged_tokens[k][0] # Conversational Target
                                    
                                    edges[(char_A, char_B)] += 1
                                    break 

    # 3. Add edges to the networkx graph
    for (u, v), weight in edges.items():
        G.add_edge(u, v, weight=weight)
        
    # Filter out isolated nodes for analysis (nodes with no connections/appearances)
    G_filtered = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()
    
    return G_filtered

# --- Temporal Metrics Function (unchanged) ---

def calculate_kanda_metrics(G, graph_type, kanda_name):
    """Calculates size, density, and top centrality for a Kanda-specific graph."""
    
    metrics = {
        'kanda': kanda_name,
        'type': graph_type,
        'active_nodes_count': len(G.nodes()),
        'edges_count': len(G.edges()),
        'density': nx.density(G) if len(G.nodes()) > 1 else 0.0,
        'centrality_scores': {} # Store all eigenvector scores for fractal analysis
    }
    
    if len(G.nodes()) > 1:
        # Calculate Eigenvector Centrality
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
        except nx.NetworkXNoConvergence:
            # Fallback to unweighted
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000) 
        except Exception:
            eigenvector_centrality = {} 
            
        metrics['centrality_scores'] = eigenvector_centrality
        
        # Get top 3 characters (only for summary)
        top_characters = sorted(
            eigenvector_centrality.items(), 
            key=lambda item: item[1], 
            reverse=True
        )[:3]
        metrics['top_centrality_summary'] = [{'character': char, 'eigenvector_centrality': score} for char, score in top_characters]

        # Track protagonist scores (Legacy tracking)
        for protag in PROTAGONISTS_TO_TRACK:
            metrics[f'{protag}_centrality'] = eigenvector_centrality.get(protag, 0.0)
            
    else:
        for protag in PROTAGONISTS_TO_TRACK:
            metrics[f'{protag}_centrality'] = 0.0

    return metrics

# --- Fractal Protagonist Analysis (Logic remains in specialized function) ---

def calculate_fractal_protagonists_with_scores(temporal_data, graph_type='undirected'):
    """Calculates the fractal protagonist score (name and score tuple) based on average normalized rank across Kandas."""
    
    # Filter data for the chosen graph type
    filtered_data = [d for d in temporal_data if d['type'] == graph_type]
    
    # Store normalized ranks for every character across all Kandas
    character_ranks = defaultdict(list)
    
    for kanda_metrics in filtered_data:
        active_nodes_count = kanda_metrics['active_nodes_count']
        scores = kanda_metrics['centrality_scores']
        
        if active_nodes_count <= 1:
            continue
            
        ranked_chars = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        char_to_rank = {char: rank + 1 for rank, (char, score) in enumerate(ranked_chars)}
        
        for char, rank in char_to_rank.items():
            normalized_score = (active_nodes_count - rank) / active_nodes_count
            character_ranks[char].append(normalized_score)

    # 3. Calculate Fractal Score (Average Normalized Score)
    fractal_scores = {}
    total_kandas = len(filtered_data)
    for char, scores in character_ranks.items():
        if len(scores) >= total_kandas / 2: 
            fractal_scores[char] = sum(scores) / len(scores)

    # 4. Get top N fractal protagonists (name, score)
    top_fractal = sorted(
        fractal_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:TOP_N_FRACTAL_PROTAGONISTS]
    
    return top_fractal # Returns list of (name, score) tuples

def plot_fractal_protagonists(fractal_scores_with_scores, graph_type):
    """Generates a horizontal bar chart of the top fractal protagonists."""
    
    characters = [item[0] for item in fractal_scores_with_scores]
    scores = [item[1] for item in fractal_scores_with_scores]
    
    plt.figure(figsize=(15, 10))
    plt.barh(characters[::-1], scores[::-1], color='maroon')
    
    plt.xlabel('Fractal Protagonist Score (Average Normalized Eigenvector Rank)', fontsize=14)
    plt.title(f'Top {TOP_N_FRACTAL_PROTAGONISTS} Fractal Protagonists ({graph_type.capitalize()} Network)', fontsize=16)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    filename = f'fractal_protagonists_{graph_type}_network.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n✅ Fractal Protagonist bar chart saved as: {filename}")

def plot_temporal_evolution(temporal_data, fractal_protagonists_names):
    """
    Generates a line plot showing the centrality evolution of the Top N Fractal Protagonists 
    across Kandas (Static Line Plot).
    """
    
    # Use the first N fractal names for the plot
    protagonists_to_plot = fractal_protagonists_names[:LINE_PLOT_N_PROTAGONISTS]
    
    # Extract the kanda names and filter data for undirected graph for plotting base
    kanda_sequence = [d['kanda'] for d in temporal_data if d['type'] == 'undirected']
    
    plt.figure(figsize=(15, 8))
    
    for protag in protagonists_to_plot:
        # Extract centrality scores for the current protagonist for all Kandas
        scores = [d['centrality_scores'].get(protag, 0.0) for d in temporal_data if d['type'] == 'undirected']
        
        plt.plot(kanda_sequence, scores, marker='o', linestyle='-', label=protag, linewidth=2)

    plt.title(f'Evolution of Centrality for Top {LINE_PLOT_N_PROTAGONISTS} Fractal Protagonists', fontsize=16)
    plt.xlabel('Ramayana Kanda (Time Step)', fontsize=14)
    plt.ylabel('Eigenvector Centrality Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Protagonist', fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = 'temporal_centrality_evolution_fractal.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n✅ Temporal evolution plot saved as: {filename}")

# --- CSV EXPORT FUNCTION (unchanged) ---

def export_kanda_networks_to_csv(text_data, all_character_set):
    """Builds and exports the node and edge lists for every Kanda network to CSV files."""
    
    print("\n--- Exporting Kanda Networks to CSV ---")
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)

    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda', 'Unknown Kanda').replace(' ', '_')

        for graph_type in ['undirected', 'directed']:
            G = build_graph_for_kanda(chapter_data, all_character_set, graph_type)
            
            if len(G.nodes()) <= 1:
                continue

            # Calculate centrality (for Node Attributes CSV)
            if graph_type == 'undirected':
                centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            else:
                # For directed, we include In-Degree as well
                centrality = nx.eigenvector_centrality(G.to_undirected(), weight='weight', max_iter=1000)
                in_degree = nx.in_degree_centrality(G)
            
            # 1. Export Edges List
            df_edges = pd.DataFrame(G.edges(data=True), columns=['Source', 'Target', 'Attributes'])
            df_edges['Weight'] = df_edges['Attributes'].apply(lambda x: x.get('weight', 1))
            df_edges = df_edges.drop(columns=['Attributes'])
            if graph_type == 'directed':
                 df_edges.rename(columns={'Source': 'Source', 'Target': 'Target'}, inplace=True)

            edges_filename = os.path.join(CSV_DIR, f'{kanda_name}_{graph_type}_edges.csv')
            df_edges.to_csv(edges_filename, index=False)
            
            # 2. Export Nodes List
            node_data = []
            for node in G.nodes():
                data = {
                    'Id': node,
                    'Eigenvector_Centrality': centrality.get(node, 0.0),
                    'Active_In_Kanda': True
                }
                if graph_type == 'directed':
                    data['In_Degree_Centrality'] = in_degree.get(node, 0.0)
                
                node_data.append(data)
                
            df_nodes = pd.DataFrame(node_data)
            nodes_filename = os.path.join(CSV_DIR, f'{kanda_name}_{graph_type}_nodes.csv')
            df_nodes.to_csv(nodes_filename, index=False)
            
            print(f"  Exported {kanda_name} ({graph_type}) - Edges/Nodes.")
    
    print("✅ All Kanda networks exported to CSV directory.")

# --- UPDATED: GIF GENERATION FUNCTION (Dynamic Cumulative Evolution) ---
def generate_fractal_gif(text_data, fractal_protagonists_list, graph_type):
    """Generates frames showing the cumulative growth of the fractal protagonist network."""
    
    print(f"\n--- Generating Cumulative Temporal GIF ({graph_type.capitalize()} Fractal Network) ---")
    
    if not os.path.exists(TEMP_FRAME_DIR):
        os.makedirs(TEMP_FRAME_DIR)
        
    # 1. Build Master Layout Graph (Aggregate Network of only Fractal Protagonists across ALL Kandas)
    G_master = nx.Graph() if graph_type == 'undirected' else nx.DiGraph()
    G_master.add_nodes_from(fractal_protagonists_list)
    
    # Pre-build all Kanda graphs for the fractal set and find master max score
    all_kanda_graphs = []
    master_max_score = 0
    
    for chapter_data in text_data['chapters']:
        G_kanda = build_graph_for_kanda(chapter_data, set(fractal_protagonists_list), graph_type)
        all_kanda_graphs.append(G_kanda)
        
        # Add to Master Graph for Layout calculation
        G_master.add_edges_from(G_kanda.edges(data=True))

        try:
            kanda_centrality = nx.eigenvector_centrality(G_kanda.to_undirected(), weight='weight', max_iter=1000)
            current_max = max(kanda_centrality.values()) if kanda_centrality else 0
            master_max_score = max(master_max_score, current_max)
        except Exception:
            pass
            
    if master_max_score == 0: master_max_score = 1

    # Calculate FIXED LAYOUT based on the aggregate network
    fixed_pos = nx.spring_layout(G_master.to_undirected(), k=0.1, iterations=100, seed=42)
    
    
    # 2. Iterate through Kandas and Generate Frames (Cumulative)
    frame_files = []
    
    # Set fixed size limits for scaling consistency across all frames
    MAX_NODE_SIZE = 5000
    MIN_NODE_SIZE = 500
    
    G_cumulative = nx.DiGraph() if graph_type == 'directed' else nx.Graph()
    G_cumulative.add_nodes_from(fractal_protagonists_list)
    
    for i, chapter_data in enumerate(text_data['chapters']):
        kanda_name = chapter_data.get('kanda', f'Kanda {i+1}')
        frame_filename = os.path.join(TEMP_FRAME_DIR, f'frame_{graph_type}_{i:02d}.png')
        frame_files.append(frame_filename)
        
        G_kanda_current = all_kanda_graphs[i]

        # ADD current Kanda's nodes/edges to the cumulative graph
        G_cumulative.add_edges_from(G_kanda_current.edges(data=True))
        
        # Calculate Centrality on the current CUMULATIVE graph
        G_centrality_source = G_cumulative.to_undirected() if graph_type == 'directed' else G_cumulative
        try:
            cumulative_centrality = nx.eigenvector_centrality(G_centrality_source, weight='weight', max_iter=1000)
        except Exception:
            cumulative_centrality = {node: 0.0 for node in G_cumulative.nodes()}
            
        
        # Calculate Node Sizes (based on cumulative centrality)
        node_sizes = []
        for node in G_cumulative.nodes():
            score = cumulative_centrality.get(node, 0)
            size = MIN_NODE_SIZE + (MAX_NODE_SIZE - MIN_NODE_SIZE) * (score / master_max_score)
            node_sizes.append(size)

        # Draw Frame
        plt.figure(figsize=(12, 10))
        plt.title(f'Cumulative Network: {kanda_name} ({graph_type.capitalize()})', fontsize=16)

        # Draw nodes using FIXED POSITION
        nx.draw_networkx_nodes(
            G_cumulative, fixed_pos, 
            node_size=node_sizes, 
            node_color='lightcoral', 
            alpha=0.9, 
            linewidths=1.0, 
            edgecolors='maroon'
        )

        # Draw edges (Dynamic thickness based on cumulative weight)
        if G_cumulative.edges():
            max_edge_weight = max([d.get('weight', 1) for u,v,d in G_cumulative.edges(data=True)])
            
            nx.draw_networkx_edges(
                G_cumulative, fixed_pos,
                width=[(G_cumulative[u][v].get('weight', 0) / max_edge_weight) * 3 + 0.5 for u, v in G_cumulative.edges()],
                alpha=0.7,
                edge_color='dimgray',
                arrows=(graph_type == 'directed'),
                arrowsize=10
            )

        # Draw labels (for all fractal protagonists)
        labels = {name: name for name in fractal_protagonists_list}
        nx.draw_networkx_labels(G_cumulative, fixed_pos, labels, font_size=10, font_weight='bold')
        
        plt.axis('off')
        plt.savefig(frame_filename, dpi=150)
        plt.close()
        print(f"  Frame generated: {kanda_name}")

    # 3. Assemble GIF
    gif_filename = GIF_FILENAME_UNDIRECTED if graph_type == 'undirected' else GIF_FILENAME_DIRECTED
    print(f"\nAssembling GIF: {gif_filename}...")
    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(gif_filename, images, duration=GIF_FRAME_DURATION)
    print(f"✅ GIF successfully created as: {gif_filename}")

    # 4. Cleanup (Done once at the end of main)
    return gif_filename

# --- Main Execution (updated for cleaner code structure) ---

def main():
    print("--- Starting Ramayana Temporal Network Analysis ---")
    try:
        text_data, character_set = load_data()
    except Exception:
        return

    # 1. KANDA-WISE METRICS COLLECTION & TEMPORAL DATA GATHERING
    temporal_metrics = []
    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda')
        if not kanda_name: continue 

        G_undirected = build_graph_for_kanda(chapter_data, character_set, 'undirected')
        metrics_undirected = calculate_kanda_metrics(G_undirected, 'undirected', kanda_name)
        temporal_metrics.append(metrics_undirected)

        G_directed = build_graph_for_kanda(chapter_data, character_set, 'directed')
        metrics_directed = calculate_kanda_metrics(G_directed, 'directed', kanda_name)
        temporal_metrics.append(metrics_directed)

    # 2. Save all temporal metrics to JSON
    with open(OUTPUT_ANALYSIS_DATA, 'w') as f:
        json.dump(temporal_metrics, f, indent=4)
        
    print(f"\n✅ Temporal metrics saved to: {OUTPUT_ANALYSIS_DATA}")
    
    # 3. FRACTAL PROTAGONIST IDENTIFICATION & PLOTTING
    
    # Defining helper function inside main to avoid global scope complexity
    def calculate_fractal_protagonists_with_scores(temporal_data, graph_type='undirected'):
        filtered_data = [d for d in temporal_data if d['type'] == graph_type]
        character_ranks = defaultdict(list)
        for kanda_metrics in filtered_data:
            active_nodes_count = kanda_metrics['active_nodes_count']
            scores = kanda_metrics['centrality_scores']
            if active_nodes_count <= 1: continue
            ranked_chars = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            char_to_rank = {char: rank + 1 for rank, (char, score) in enumerate(ranked_chars)}
            for char, rank in char_to_rank.items():
                normalized_score = (active_nodes_count - rank) / active_nodes_count
                character_ranks[char].append(normalized_score)

        fractal_scores = {}
        total_kandas = len(filtered_data)
        for char, scores in character_ranks.items():
            if len(scores) >= total_kandas / 2: 
                fractal_scores[char] = sum(scores) / len(scores)

        top_fractal = sorted(
            fractal_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )[:TOP_N_FRACTAL_PROTAGONISTS]
        
        return top_fractal 

    undirected_fractal_protagonists_with_scores = calculate_fractal_protagonists_with_scores(temporal_metrics, 'undirected')
    directed_fractal_protagonists_with_scores = calculate_fractal_protagonists_with_scores(temporal_metrics, 'directed')
    
    undirected_fractal_protagonists_names = [name for name, score in undirected_fractal_protagonists_with_scores]
    directed_fractal_protagonists_names = [name for name, score in directed_fractal_protagonists_with_scores]


    # 4. TEMPORAL VISUALIZATIONS & PLOTTING
    
    # Line Plot: Show evolution of Top 10 Fractal Protagonists' centrality
    plot_temporal_evolution(temporal_metrics, undirected_fractal_protagonists_names)

    # GIF Plot (Dynamic Cumulative Fractal Network)
    generate_fractal_gif(text_data, undirected_fractal_protagonists_names, 'undirected')
    generate_fractal_gif(text_data, directed_fractal_protagonists_names, 'directed')
    
    # Bar Chart Plotting
    print(f"\n--- Top {TOP_N_FRACTAL_PROTAGONISTS} Fractal Protagonists (Undirected) ---")
    for rank, (name, score) in enumerate(undirected_fractal_protagonists_with_scores):
        print(f"Rank {rank+1}: {name} (Score: {score:.4f})")
    plot_fractal_protagonists(undirected_fractal_protagonists_with_scores, 'undirected')

    print(f"\n--- Top {TOP_N_FRACTAL_PROTAGONISTS} Fractal Protagonists (Directed) ---")
    for rank, (name, score) in enumerate(directed_fractal_protagonists_with_scores):
        print(f"Rank {rank+1}: {name} (Score: {score:.4f})")
    plot_fractal_protagonists(directed_fractal_protagonists_with_scores, 'directed')
    
    # 5. CSV EXPORT
    export_kanda_networks_to_csv(text_data, character_set)
    
    # 6. Final Cleanup
    if os.path.exists(TEMP_FRAME_DIR):
        try:
            # Re-running cleanup in case GIF generation failed mid-process
            for f in os.listdir(TEMP_FRAME_DIR):
                os.remove(os.path.join(TEMP_FRAME_DIR, f))
            os.rmdir(TEMP_FRAME_DIR)
        except OSError:
             print("Warning: Could not completely remove temporary frame directory.")

    print("\n--- Temporal Analysis Complete ---")


if __name__ == '__main__':
    main()