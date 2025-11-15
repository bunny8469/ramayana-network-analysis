import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os 
import numpy as np
import unidecode 
# NOTE: VADER must be installed separately: pip install vaderSentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    class SentimentIntensityAnalyzer:
        """Placeholder for VADER if not installed."""
        def polarity_scores(self, sentence):
            # This is a dummy score if VADER is not present.
            return {'compound': 0.0} 

# NOTE: Pyvis must be installed separately: pip install pyvis
from pyvis.network import Network 

# --- Configuration (Based on previous analysis) ---
FILE_TEXT = './../data/ramayana_resolved_final.json'
FILE_CANONICAL_NAMES = './../data/canonical_characters.json'
CO_OCCURRENCE_WINDOW = 15 
TOP_N_VIZ_NODES = 800 # MODIFIED: Increased limit to ensure ALL active nodes (~777 total) are included in the visualization.

# --- Sentiment Configuration ---
SENTIMENT_THRESHOLD = 0.05 # VADER standard threshold for polarity interpretation

# Visualization & Output Configuration
NODE_SIZE_SCALER = 40 # MODIFIED: Increased base size for Pyvis nodes (from 20 to 40)
OUTPUT_SENTIMENT_DIR = 'sentiment_plots/'
TRAJECTORY_THRESHOLD = 0.10 # Minimum swing for VADER compound score trajectory visualization

# --- Matplotlib Plotting Limits ---
PLOTS_PER_PAGE = 16 
SUBPLOT_COLS = 4
SUBPLOT_ROWS = 4
FIGURE_WIDTH = 8 * SUBPLOT_COLS
FIGURE_HEIGHT = 5 * SUBPLOT_ROWS


# Discourse markers for conversational graph (Ignored in this analysis)
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- VADER Scoring Implementation (Based on User's Documentation) ---

SID_OBJ = SentimentIntensityAnalyzer()
CONTEXT_WINDOW = 20 

def get_vader_scores(sentence):
    """Returns the compound sentiment score for a given sentence."""
    return SID_OBJ.polarity_scores(sentence)['compound']

# --- Core Utility Functions (Updated to remove Directed logic) ---

def load_data():
    """Loads text and canonical names."""
    with open(FILE_TEXT, 'r') as f:
        text_data = json.load(f)
    with open(FILE_CANONICAL_NAMES, 'r') as f:
        canonical_names = json.load(f)
    return text_data, set(canonical_names)

def tokenize_and_tag(canto_text, character_set):
    """Tokenizes text and tags words as characters."""
    words = re.findall(r'\b\w+\b', canto_text.strip())
    tagged_tokens = []
    for word in words:
        is_char = word in character_set
        tagged_tokens.append((word, is_char))
    return tagged_tokens

def calculate_kanda_metrics(G, kanda_name, graph_type='undirected'):
    """Calculates centrality for a Kanda-specific graph."""
    metrics = {'kanda': kanda_name, 'type': graph_type, 'centrality_scores': {}, 'active_nodes_count': len(G.nodes())}
    if len(G.nodes()) > 1:
        # Use 'count' for weight, as this is an undirected co-occurrence graph.
        weight_attr = 'count'
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, weight=weight_attr, max_iter=1000)
        except Exception:
            eigenvector_centrality = {}
        metrics['centrality_scores'] = eigenvector_centrality
    return metrics

# The Undirected Sentiment calculation function is now the primary and only one needed for scoring
def analyze_sentiment_interactions_kanda_wise(text_data, all_character_set, graph_type='undirected'):
    """
    Scans text using VADER logic for UNDIRECTED co-occurrence (shared context) 
    and returns SUMMED compound scores structured by Kanda.
    """
    all_kanda_raw_scores = {}
    
    for chapter in text_data['chapters']:
        kanda_name = chapter.get('kanda')
        if not kanda_name: continue
        
        kanda_edge_data = defaultdict(lambda: {'sum_score': 0.0, 'count': 0})
        canto_texts = chapter.get('cantos', [])
        
        for canto_text in canto_texts:
            clean_text = re.sub(r'[^a-zA-Z\s]', ' ', canto_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
            words = clean_text.split()
            
            # Use the entire character set since we are not filtering by fractal protagonists yet
            for i, word in enumerate(words):
                if word.capitalize() in all_character_set or word in all_character_set: 
                    char_A = word.capitalize() 
                    
                    # Define context window: symmetric around the current word.
                    start = max(0, i - CO_OCCURRENCE_WINDOW // 2)
                    end = min(len(words), i + CO_OCCURRENCE_WINDOW // 2)
                    context_words = words[start:end]
                    context_text = " ".join(context_words)
                    
                    # 1. Score the context using VADER compound score
                    compound_score = get_vader_scores(context_text)
                    
                    # 2. Identify targets within the context window
                    targets_in_context = set()
                    for c_word in context_words: 
                        if c_word.capitalize() in all_character_set or c_word in all_character_set:
                            targets_in_context.add(c_word.capitalize())
                            
                    # 3. Aggregate score (strictly undirected)
                    for char_B in targets_in_context:
                        if char_A != char_B:
                            # Undirected: canonical ordering (min, max)
                            u, v = tuple(sorted((char_A, char_B)))
                            
                            kanda_edge_data[(u, v)]['sum_score'] += compound_score
                            kanda_edge_data[(u, v)]['count'] += 1

        # Final structure for Kanda sentiment data
        kanda_final_edges = {}
        for (u, v), data in kanda_edge_data.items():
            avg_score = data['sum_score'] / data['count'] if data['count'] > 0 else 0.0
            
            kanda_final_edges[(u, v)] = {
                'count': data['count'], 
                'sentiment': avg_score,
            }
            
        all_kanda_raw_scores[kanda_name] = kanda_final_edges
        
    return all_kanda_raw_scores

# --- 2. Pyvis Network Generation (Shared) ---

def get_sentiment_color(sentiment_score):
    """Maps VADER compound score to color."""
    if sentiment_score >= SENTIMENT_THRESHOLD:
        return '#00CC00' # Bright Green (Positive)
    elif sentiment_score <= -SENTIMENT_THRESHOLD:
        return '#FF0000' # Bright Red (Negative)
    else:
        return '#AAAAAA' # Gray (Neutral)

def get_sentiment_title(sentiment_score):
    """Returns a descriptive title for the sentiment score."""
    if sentiment_score >= SENTIMENT_THRESHOLD:
        return "Positive"
    elif sentiment_score <= -SENTIMENT_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"

def generate_pyvis_network(G_full, centrality_scores, kanda_name, graph_type, filename):
    """Generates an interactive Pyvis network plot for the entire network."""
    
    # Filter graph to top N nodes for visualization clarity
    sorted_nodes = sorted(G_full.degree(weight='count'), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, degree in sorted_nodes[:TOP_N_VIZ_NODES]]
    
    G_viz = G_full.subgraph(top_nodes).copy()
    
    is_directed = (graph_type == 'directed')
    
    # Initialize Pyvis Network
    net = Network(height="750px", width="100%", directed=is_directed, notebook=False, 
                  heading=f"{kanda_name} Undirected Sentiment Network (Top {len(G_viz.nodes())} Active Nodes)")
    
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "springLength": 100
        },
        "minVelocity": 0.75,
        "solver": "forceAtlas2Based"
      }
    }
    """) # Use ForceAtlas2 for better cluster separation

    # 1. Add Nodes
    max_eigen = max(centrality_scores.values()) if centrality_scores else 1

    for node in G_viz.nodes():
        eigen_score = centrality_scores.get(node, 0)
        node_size = (eigen_score / max_eigen) * NODE_SIZE_SCALER + 10 # Scale size
        
        # Node Title for hover information
        node_title = f"<b>{node}</b><br>Prestige (Eigenvector): {eigen_score:.4f}<br>Total Scenes: {G_viz.degree(node, weight='count'):.0f}"

        net.add_node(
            node, 
            label=node, 
            title=node_title, 
            value=node_size, # Size by Eigenvector Centrality (Prestige)
            color='#1E90FF', # Consistent Node Color (Dodger Blue)
            font={'size': 18, 'face': 'Arial', 'color': 'black'} # MODIFIED: Increased font size to 18
        )

    # 2. Add Edges
    max_weight = max([d['count'] for u, v, d in G_viz.edges(data=True)]) if G_viz.edges() else 1

    for u, v, data in G_viz.edges(data=True):
        sentiment = data['sentiment']
        count = data['count']
        
        edge_color = get_sentiment_color(sentiment)
        edge_title = f"Avg. Sentiment: {sentiment:.3f} ({get_sentiment_title(sentiment)})<br>Co-occurrences: {count}"
        edge_width = (count / max_weight) * 3 + 1 # Width by Interaction Count

        net.add_edge(
            u, v, 
            value=edge_width, # Width by Interaction Volume
            title=edge_title, # Hover text
            color=edge_color, # Color by Sentiment
            arrows='to' if is_directed else None, 
            arrowStrikethrough=False
        )

    # Save as HTML
    net.show(filename, notebook=False)
    print(f"✅ Interactive Pyvis plot saved as: {filename}")


# --- 3. Relationship Trajectory Plot (Matplotlib kept for line charts) ---

def plot_pair_sentiment_trajectory(kanda_names, all_sentiment_scores_by_pair, graph_type):
    """Identifies pairs with major sentiment shifts and plots their trajectory."""
    
    print(f"\n--- Analyzing Relationship Trajectories ({graph_type.capitalize()} Network) ---")
    
    # 1. Find the Top 16 Most Active Pairs overall
    all_pairs_volume = defaultdict(float)
    # The input format is {pair: {'scores': [...], 'counts': [...], 'kanda_order': [...]}}
    for pair, data in all_sentiment_scores_by_pair.items():
        all_pairs_volume[pair] = sum(data['counts']) # CORRECTED ACCESS

    top_active_pairs = sorted(all_pairs_volume.items(), key=lambda item: item[1], reverse=True)[:PLOTS_PER_PAGE]
    
    # Filter the trajectory data to only include these top pairs
    pairs_to_plot = {} 
    
    for pair, volume in top_active_pairs:
        data = all_sentiment_scores_by_pair[pair]
        
        scores = data['scores']
        kandas = data['kanda_order']
        
        # Ensure the pair has enough data and check for a significant swing
        sentiment_swing = max(scores) - min(scores)
        
        if sentiment_swing >= TRAJECTORY_THRESHOLD:
            # Reformat for plotting: list of (kanda, score) tuples
            pairs_to_plot[pair] = list(zip(kandas, scores))
            
    if not pairs_to_plot:
        print(f"No character pairs found with a significant sentiment swing (> {TRAJECTORY_THRESHOLD}) among the top active pairs.")
        return

    # --- Plotting in pages ---
    
    pair_list = list(pairs_to_plot.items())
    num_pairs = len(pair_list)
    
    for page_num in range(0, num_pairs, PLOTS_PER_PAGE):
        start_index = page_num
        end_index = min(page_num + PLOTS_PER_PAGE, num_pairs)
        current_pairs = pair_list[start_index:end_index]
        
        plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
        
        print(f"Visualizing Page {page_num // PLOTS_PER_PAGE + 1} ({len(current_pairs)} pairs)...")

        for i, ((u, v), scores) in enumerate(current_pairs):
            plt.subplot(SUBPLOT_ROWS, SUBPLOT_COLS, i + 1)
            
            # scores is already a list of (kanda_name, score) tuples
            kanda_indices = [kanda_names.index(kanda) for kanda, score in scores]
            kanda_labels = [kanda for kanda, score in scores]
            score_values = [score for kanda, score in scores]
            
            plt.plot(kanda_indices, score_values, marker='o', linestyle='-', linewidth=2, alpha=0.7, color='blue')
            
            start_score = score_values[0]
            end_score = score_values[-1]
            
            # Plot markers based on VADER polarity interpretation
            plt.plot(kanda_indices[0], start_score, marker='o', 
                     color='green' if start_score >= SENTIMENT_THRESHOLD else ('red' if start_score <= -SENTIMENT_THRESHOLD else 'gray'), 
                     markersize=8)
            plt.plot(kanda_indices[-1], end_score, marker='o', 
                     color='green' if end_score >= SENTIMENT_THRESHOLD else ('red' if end_score <= -SENTIMENT_THRESHOLD else 'gray'), 
                     markersize=8, label='Final Score')
            
            # Draw the neutral/threshold line
            plt.axhspan(-SENTIMENT_THRESHOLD, SENTIMENT_THRESHOLD, color='gray', alpha=0.1, label='Neutral Zone')
            plt.axhline(0, color='black', linestyle=':', linewidth=1)
            
            plt.title(f'{u} - {v}', fontsize=12) # Undirected clarity
            plt.xticks(kanda_indices, kanda_labels, rotation=45, ha='right', fontsize=8)
            plt.ylabel('Avg. VADER Compound Score', fontsize=10)
            plt.ylim(-1.0, 1.0) # Consistent Y-axis range for VADER compound score
            plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        filename = os.path.join(OUTPUT_SENTIMENT_DIR, f'sentiment_relationship_trajectories_{graph_type}_page_{page_num // PLOTS_PER_PAGE + 1}.png')
        plt.savefig(filename, dpi=200) # Reduced DPI slightly for faster rendering
        plt.close()
        print(f"✅ Trajectory plot page {page_num // PLOTS_PER_PAGE + 1} saved as: {filename}")


# --- Main Execution ---

def main():
    print("--- Starting Undirected Sentiment Analysis (VADER/Pyvis) ---")
    try:
        text_data, all_character_set = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not os.path.exists(OUTPUT_SENTIMENT_DIR):
        os.makedirs(OUTPUT_SENTIMENT_DIR)
    
    # --- 1. CALCULATE SENTIMENT FOR UNDIRECTED (Co-occurrence) NETWORK (ENTIRE NETWORK) ---
    
    # We use the full character set for analysis
    undirected_sentiment_data = analyze_sentiment_interactions_kanda_wise(text_data, all_character_set, 'undirected')
    
    # Aggregate data across all Kandas for the Overall/Master plot 
    undirected_overall_raw_scores = defaultdict(lambda: defaultdict(list)) # Stores {pair: {kanda: [score, count]}}
    kanda_names_list = []
    
    for kanda, edges in undirected_sentiment_data.items():
        if kanda not in kanda_names_list: kanda_names_list.append(kanda)
        for (u, v), data in edges.items():
            # Store scores by pair, preserving kanda order
            undirected_overall_raw_scores[(u, v)]['scores'].append(data['sentiment']) 
            undirected_overall_raw_scores[(u, v)]['counts'].append(data['count']) 
            undirected_overall_raw_scores[(u, v)]['kanda_order'].append(kanda) 


    # --- 2. UNDIRECTED PLOTS GENERATION (PYVIS) ---

    # Build Overall Undirected Graph
    undirected_overall_edges = {}
    for (u, v), data in undirected_overall_raw_scores.items():
        avg_score = np.mean(data['scores'])
        total_count = sum(data['counts'])
        undirected_overall_edges[(u, v)] = {'count': total_count, 'sentiment': avg_score}
    
    G_undirected_overall = nx.Graph()
    G_undirected_overall.add_nodes_from(all_character_set)
    for (u, v), data in undirected_overall_edges.items():
        G_undirected_overall.add_edge(u, v, **data)
        
    G_undirected_full = G_undirected_overall.subgraph([n for n in G_undirected_overall.nodes() if G_undirected_overall.degree(n) > 0]).copy()
    
    # Calculate centrality on the full graph
    overall_eigenvector_undirected = nx.eigenvector_centrality(G_undirected_full, weight='count', max_iter=1000)
    
    # Generate Overall Undirected Pyvis plot (Filtered to TOP_N_VIZ_NODES)
    generate_pyvis_network(G_undirected_full, overall_eigenvector_undirected, "Overall", 'undirected', os.path.join(OUTPUT_SENTIMENT_DIR, 'sentiment_overall_undirected_network.html'))

    # Kanda-wise Undirected Plots
    for kanda_name, edges in undirected_sentiment_data.items():
        if not edges: continue
        G_kanda = nx.Graph()
        G_kanda.add_nodes_from(all_character_set)
        for (u, v), data in edges.items():
             G_kanda.add_edge(u, v, **data)
        G_kanda_full = G_kanda.subgraph([n for n in G_kanda.nodes() if G_kanda.degree(n) > 0]).copy()
        kanda_eigenvector = nx.eigenvector_centrality(G_kanda_full, weight='count', max_iter=1000)
        generate_pyvis_network(G_kanda_full, kanda_eigenvector, kanda_name, 'undirected', os.path.join(OUTPUT_SENTIMENT_DIR, f'sentiment_{kanda_name.replace(" ", "_")}_undirected_network.html'))


    # --- 3. RELATIONSHIP TRAJECTORY PLOT (MATPLOTLIB) ---
    
    # Reformat scores for the trajectory plot function
    # NOTE: The raw scores dictionary already contains all the necessary data organized by pair.
    # We pass the raw data structure: undirected_overall_raw_scores
    
    # Plotting the sentiment trajectory for the UNDIRECTED Network
    plot_pair_sentiment_trajectory(kanda_names_list, undirected_overall_raw_scores, 'undirected')

    print("\n--- Final Sentiment Analysis Summary ---")
    print(f"All sentiment networks (Overall + 7 Kandas) are saved as interactive HTML files in the '{OUTPUT_SENTIMENT_DIR}' directory.")
    print(f"The relationship trajectory plot (line chart) for the UNDIRECTED network is saved as a PNG.")


if __name__ == '__main__':
    main()