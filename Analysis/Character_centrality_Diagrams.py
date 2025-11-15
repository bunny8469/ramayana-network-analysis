import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import numpy as np
import os 

# --- Configuration (Copied from temporal analysis for consistency) ---
FILE_TEXT = './../data/ramayana_resolved_final.json'
FILE_CANONICAL_NAMES = './../data/canonical_characters.json'
CO_OCCURRENCE_WINDOW = 15 
TOP_N_FRACTAL_PROTAGONISTS = 30 # Number of characters to plot on each CCD
TOP_N_TRAJECTORY_CHARS = 10 # Number of characters to track across the final temporal CCD

# CCD Aesthetics
DOT_SIZE_BASE = 100 # Increased base size for all dots
DOT_SIZE_MULTIPLIER = 12000 # Increased multiplier for scaling contrast
OUTPUT_CCD_DIR = 'ccd_visualizations/'

# Visualization Enhancements
NODE_ALPHA = 0.2 # Transparency of the nodes
LABEL_FONT_SIZE = 16
LABEL_FONT_WEIGHT = 'bold'
TRAJECTORY_LINE_WIDTH = 3 # Increased line width for clarity
TRAJECTORY_MARKER_SIZE = 11 # Increased marker size for clarity


# Discourse markers for conversational graph
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- Core Utility Functions (Needed for network building and fractal ranking) ---

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

def build_directed_graph_for_kanda(kanda_data, character_set):
    """Builds a Directed (Conversational) Graph based on a Kanda's text."""
    
    G = nx.DiGraph()
    G.add_nodes_from(character_set)
    edges = Counter()
    
    canto_texts = kanda_data.get('cantos', [])
    
    for canto_text in canto_texts:
        tagged_tokens = tokenize_and_tag(canto_text, character_set)
        
        for i, (word, is_char) in enumerate(tagged_tokens):
            if is_char:
                char_A = word 
                for j in range(i + 1, min(i + CO_OCCURRENCE_WINDOW, len(tagged_tokens))):
                    if DISCOURSE_MARKER_PATTERN.search(tagged_tokens[j][0]):
                        for k in range(j + 1, min(j + 5, len(tagged_tokens))):
                            if tagged_tokens[k][1] and tagged_tokens[k][0] != char_A:
                                char_B = tagged_tokens[k][0] 
                                edges[(char_A, char_B)] += 1
                                break 

    for (u, v), weight in edges.items():
        G.add_edge(u, v, weight=weight)
        
    G_filtered = G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()
    
    return G_filtered

# --- Fractal Protagonist Identification (Used to filter final plots) ---

def calculate_fractal_protagonists_names(temporal_data):
    """Calculates the top fractal protagonists' names from the directed network."""
    
    filtered_data = [d for d in temporal_data if d['type'] == 'directed']
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
    
    return [name for name, score in top_fractal] 

# --- CCD Specific Analysis and Plotting ---

def calculate_ccd_metrics(G_directed):
    """Calculates PageRank (X), Eigenvector (Y), and Weighted Degree (Size/Color)."""
    
    if len(G_directed.nodes()) <= 1:
        return {}

    # 1. PageRank Centrality (X-axis: Agency)
    pagerank_centrality = nx.pagerank(G_directed, weight='weight', alpha=0.85)

    # 2. Eigenvector Centrality (Y-axis: Prestige/Influence)
    # Use the undirected projection for stability, as PageRank already handles directionality well.
    G_undirected = G_directed.to_undirected()
    try:
        eigenvector_centrality = nx.eigenvector_centrality(G_undirected, weight='weight', max_iter=1000)
    except nx.NetworkXNoConvergence:
        eigenvector_centrality = {node: 0.0 for node in G_undirected.nodes()} 

    # 3. Weighted Degree Centrality (Size/Color: Interactions)
    weighted_degree = dict(G_directed.degree(weight='weight'))

    ccd_data = {}
    for node in G_directed.nodes():
        if node in pagerank_centrality and node in eigenvector_centrality:
            ccd_data[node] = {
                'pagerank': pagerank_centrality.get(node, 0.0),
                'eigenvector': eigenvector_centrality.get(node, 0.0),
                'weighted_degree': weighted_degree.get(node, 0.0)
            }
    return ccd_data

def plot_single_ccd(ccd_data, fractal_protagonists, kanda_name, max_weighted_degree):
    """Generates a Character Centrality Diagram (CCD) for a single Kanda."""
    
    if not ccd_data: return

    # Filter data to only include the fractal protagonists
    plot_data = {
        name: data for name, data in ccd_data.items() 
        if name in fractal_protagonists
    }

    if not plot_data: return

    names = list(plot_data.keys())
    x = [plot_data[name]['pagerank'] for name in names]
    y = [plot_data[name]['eigenvector'] for name in names]
    s = [DOT_SIZE_BASE + (plot_data[name]['weighted_degree'] / max_weighted_degree) * DOT_SIZE_MULTIPLIER for name in names]
    c = [plot_data[name]['weighted_degree'] for name in names] # Color by weighted degree

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(x, y, s=s, c=c, cmap='viridis', alpha=NODE_ALPHA, edgecolors='black', linewidths=1.5)
    
    # Label the characters
    for i, name in enumerate(names):
        plt.annotate(
            name, 
            (x[i], y[i]),
            textcoords="offset points", 
            xytext=(5, 5),
            ha='left',
            fontsize=LABEL_FONT_SIZE,
            fontweight=LABEL_FONT_WEIGHT
        )

    plt.xlabel('PageRank Centrality (Agency)', fontsize=14)
    plt.ylabel('Eigenvector Centrality (Prestige)', fontsize=14)
    plt.title(f'Character Centrality Diagram: {kanda_name}', fontsize=16)
    
    # Add color bar for weighted degree
    cbar = plt.colorbar(scatter, label='Weighted Degree Centrality (Total Interactions)')
    
    # Add Legend for node size (approximation based on max/min size)
    legend_elements = [
        plt.scatter([], [], s=DOT_SIZE_BASE + 0.1 * DOT_SIZE_MULTIPLIER, c='gray', alpha=NODE_ALPHA, edgecolors='black', label='Low Interactions'),
        plt.scatter([], [], s=DOT_SIZE_BASE + 0.5 * DOT_SIZE_MULTIPLIER, c='gray', alpha=NODE_ALPHA, edgecolors='black', label='Medium Interactions'),
        plt.scatter([], [], s=DOT_SIZE_BASE + 0.9 * DOT_SIZE_MULTIPLIER, c='gray', alpha=NODE_ALPHA, edgecolors='black', label='High Interactions')
    ]
    plt.legend(handles=legend_elements, title='Weighted Degree (Size)', loc='upper left')

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    filename = os.path.join(OUTPUT_CCD_DIR, f'ccd_{kanda_name.replace(" ", "_")}.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"  CCD plot generated: {filename}")
    
    # Return the data for the trajectory plot
    return plot_data

def plot_ccd_trajectory(all_ccd_data_by_kanda):
    """Plots the temporal trajectory of the top protagonists across the CCD plane."""
    
    # 1. Identify top 5 overall central characters to track
    all_kanda_names = list(all_ccd_data_by_kanda.keys())
    
    # Find overall average Eigenvector centrality across all Kandas for stable tracking
    avg_eigen_scores = defaultdict(float)
    char_kanda_count = defaultdict(int)
    
    for kanda_name, ccd_data in all_ccd_data_by_kanda.items():
        for name, metrics in ccd_data.items():
            avg_eigen_scores[name] += metrics['eigenvector']
            char_kanda_count[name] += 1
            
    for name, total_score in avg_eigen_scores.items():
        avg_eigen_scores[name] = total_score / char_kanda_count[name]

    # Select the Top N for the trajectory plot
    top_trajectory_chars = sorted(
        avg_eigen_scores.keys(),
        key=avg_eigen_scores.get,
        reverse=True
    )[:TOP_N_TRAJECTORY_CHARS]
    
    
    # 2. Extract Trajectory Data
    trajectories = defaultdict(lambda: {'x': [], 'y': [], 'kanda': []})
    
    for kanda_index, kanda_name in enumerate(all_kanda_names):
        ccd_data = all_ccd_data_by_kanda[kanda_name]
        for char in top_trajectory_chars:
            if char in ccd_data:
                trajectories[char]['x'].append(ccd_data[char]['pagerank'])
                trajectories[char]['y'].append(ccd_data[char]['eigenvector'])
                trajectories[char]['kanda'].append(kanda_index)

    # 3. Plotting
    plt.figure(figsize=(14, 12))
    
    # Calculate global min/max for stable axes
    max_x = max(max(traj['x']) for traj in trajectories.values()) * 1.05
    max_y = max(max(traj['y']) for traj in trajectories.values()) * 1.05
    
    for char, traj in trajectories.items():
        if len(traj['x']) > 1:
            # Plot line connecting the Kanda points
            plt.plot(traj['x'], traj['y'], marker='o', linestyle='-', label=char, linewidth=TRAJECTORY_LINE_WIDTH, alpha=0.8, markersize=TRAJECTORY_MARKER_SIZE)
            
            # Label start (Kanda 1) and end (Last Kanda) points
            plt.annotate(f"Start: {char}", (traj['x'][0], traj['y'][0]), fontsize=12, xytext=(5, 0), textcoords='offset points', fontweight='bold')
            plt.annotate(f"End: {char}", (traj['x'][-1], traj['y'][-1]), fontsize=12, xytext=(5, 0), textcoords='offset points', fontweight='bold')
            
            # Plot intermediate Kanda markers (small numbers)
            for i in range(len(traj['x'])):
                plt.text(traj['x'][i], traj['y'][i], str(i+1), fontsize=10, ha='center', va='center', color='black',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.xlim(0, max_x)
    plt.ylim(0, max_y)
    plt.xlabel('PageRank Centrality (Agency)', fontsize=14)
    plt.ylabel('Eigenvector Centrality (Prestige)', fontsize=14)
    plt.title(f'Temporal Trajectory of Top {TOP_N_TRAJECTORY_CHARS} Fractal Protagonists (CCD)', fontsize=16)
    plt.legend(title='Protagonist', fontsize=12, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    filename = os.path.join(OUTPUT_CCD_DIR, 'ccd_temporal_trajectory.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n✅ Temporal CCD Trajectory plot generated: {filename}")


# --- Main Execution ---

def main():
    print("--- Starting Character Centrality Diagram (CCD) Analysis ---")
    try:
        text_data, character_set = load_data()
    except Exception:
        return

    if not os.path.exists(OUTPUT_CCD_DIR):
        os.makedirs(OUTPUT_CCD_DIR)

    # --- 1. Kanda-wise Network Metrics Collection ---
    temporal_metrics_directed = []
    
    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda')
        if not kanda_name: continue 

        G_directed = build_directed_graph_for_kanda(chapter_data, character_set)
        
        # Calculate Eigenvector (for fractal ranking)
        metrics = calculate_kanda_metrics(G_directed, 'directed', kanda_name)
        temporal_metrics_directed.append(metrics)
        
    # --- 2. Identify Fractal Protagonists ---
    # We rely on the Eigenvector scores gathered above to define the key subset of characters.
    fractal_protagonists = calculate_fractal_protagonists_names(temporal_metrics_directed)
    print(f"\nIdentified {len(fractal_protagonists)} Fractal Protagonists based on consistent centrality.")
    
    # --- 3. Generate Kanda CCDs ---
    all_ccd_data_by_kanda = {}
    
    # Find the maximum weighted degree across all Kandas for consistent size scaling
    max_weighted_degree = 0
    for chapter_data in text_data['chapters']:
        G_directed = build_directed_graph_for_kanda(chapter_data, character_set)
        weighted_degree = dict(G_directed.degree(weight='weight'))
        current_max = max(weighted_degree.values()) if weighted_degree else 0
        max_weighted_degree = max(max_weighted_degree, current_max)

    print("\n--- Generating Kanda CCD Plots ---")
    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda')
        if not kanda_name: continue 

        G_directed = build_directed_graph_for_kanda(chapter_data, character_set)
        
        ccd_metrics = calculate_ccd_metrics(G_directed)
        
        # Plot the CCD for this Kanda
        kanda_plot_data = plot_single_ccd(
            ccd_metrics, 
            fractal_protagonists, 
            kanda_name, 
            max_weighted_degree
        )
        
        # Store the raw metrics for the final trajectory plot
        if kanda_plot_data:
            all_ccd_data_by_kanda[kanda_name] = kanda_plot_data
            
    # --- 4. Generate Temporal Trajectory CCD ---
    if all_ccd_data_by_kanda:
        plot_ccd_trajectory(all_ccd_data_by_kanda)

    print("\n--- CCD Analysis Complete ---")
    print(f"All CCD plots and the temporal trajectory plot are saved in the '{OUTPUT_CCD_DIR}' directory.")


if __name__ == '__main__':
    # Helper function definition needed for CCD analysis due to circular dependencies 
    # (CCD needs fractal data, fractal data needs centrality, which we calculate here).
    def calculate_kanda_metrics(G, graph_type, kanda_name):
        metrics = {'kanda': kanda_name, 'type': graph_type, 'centrality_scores': {}}
        if len(G.nodes()) > 1:
            try:
                eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
            except nx.NetworkXNoConvergence:
                eigenvector_centrality = nx.eigenvector_centrality(G.to_undirected(), max_iter=1000) # Fallback
            except Exception:
                eigenvector_centrality = {}
            metrics['centrality_scores'] = eigenvector_centrality
            metrics['active_nodes_count'] = len(G.nodes())
        return metrics

    main()