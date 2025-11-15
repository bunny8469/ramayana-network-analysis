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
TOP_N_FRACTAL_PROTAGONISTS = 30 # Characters considered in the sentiment analysis

# --- Sentiment Configuration ---
SENTIMENT_THRESHOLD = 0.05 # VADER standard threshold for polarity interpretation

# Visualization & Output Configuration
NODE_SIZE_SCALER = 20 # Base size for Pyvis nodes
OUTPUT_SENTIMENT_DIR = 'sentiment_plots/'
TRAJECTORY_THRESHOLD = 0.10 # Minimum swing for VADER compound score trajectory visualization

# --- Matplotlib Plotting Limits ---
PLOTS_PER_PAGE = 16 
SUBPLOT_COLS = 4
SUBPLOT_ROWS = 4
FIGURE_WIDTH = 8 * SUBPLOT_COLS
FIGURE_HEIGHT = 5 * SUBPLOT_ROWS


# Discourse markers for conversational graph (Source -> Target)
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- VADER Scoring Implementation (Based on User's Documentation) ---

SID_OBJ = SentimentIntensityAnalyzer()
CONTEXT_WINDOW = 20 # Using 20 words for context around the source character

def get_vader_scores(sentence):
    """Returns the compound sentiment score for a given sentence."""
    return SID_OBJ.polarity_scores(sentence)['compound']

# --- Core Utility Functions (Unchanged) ---

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

def calculate_kanda_metrics(G, kanda_name, graph_type='directed'):
    """Calculates centrality for a Kanda-specific graph (required for fractal scoring)."""
    metrics = {'kanda': kanda_name, 'type': graph_type, 'centrality_scores': {}, 'active_nodes_count': len(G.nodes())}
    if len(G.nodes()) > 1:
        # CCD Prestige uses undirected eigenvector centrality, so we ensure the graph is compatible.
        G_centrality = G.to_undirected() if graph_type == 'directed' else G
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G_centrality, weight='weight', max_iter=1000)
        except Exception:
            eigenvector_centrality = {}
        metrics['centrality_scores'] = eigenvector_centrality
    return metrics

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
    return G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()

def build_undirected_graph_for_kanda(kanda_data, character_set):
    """Builds an Undirected (Co-occurrence) Graph based on a Kanda's text."""
    G = nx.Graph()
    G.add_nodes_from(character_set)
    co_occurrence_edges = Counter()

    canto_texts = kanda_data.get('cantos', [])
    for canto_text in canto_texts:
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
        G.add_edge(u, v, weight=weight)
        
    return G.subgraph([n for n in G.nodes() if G.degree(n) > 0]).copy()

def calculate_fractal_protagonists_names(temporal_data):
    """Calculates the top fractal protagonists' names from the directed network (used as the canonical set)."""
    
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

# --- 1. VADER-Based Sentiment Scoring Logic (New Implementation) ---

def analyze_sentiment_interactions_undirected(text_data, fractal_protagonists_set):
    """
    Scans text using VADER logic (context window) for UNDIRECTED co-occurrence and returns 
    SUMMED compound scores structured by Kanda.
    """
    all_kanda_raw_scores = {}
    
    for chapter in text_data['chapters']:
        kanda_name = chapter.get('kanda')
        if not kanda_name: continue
        
        # Raw scores stores the SUM of compound sentiment scores
        # Key is (min(U,V), max(U,V)) to ensure symmetry
        kanda_edge_data = defaultdict(lambda: {'sum_score': 0.0, 'count': 0})
        canto_texts = chapter.get('cantos', [])
        
        for canto_text in canto_texts:
            # Clean text once per canto
            clean_text = re.sub(r'[^a-zA-Z\s]', ' ', canto_text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip().lower()
            words = clean_text.split()
            
            # Use an iteration that is robust to context windows
            for i, word in enumerate(words):
                
                # Check if the word is a fractal protagonist (in canonical form)
                if word.capitalize() in fractal_protagonists_set or word in fractal_protagonists_set: 
                    char_A = word.capitalize() 
                    
                    # Define context window: center word and surrounding words
                    start = max(0, i - CONTEXT_WINDOW // 2)
                    end = min(len(words), i + CONTEXT_WINDOW // 2)
                    context_words = words[start:end]
                    context_text = " ".join(context_words)
                    
                    # 1. Score the context using VADER compound score
                    compound_score = get_vader_scores(context_text)
                    
                    # 2. Identify targets within the context window
                    targets_in_context = set()
                    for c_word in context_words: 
                        if c_word.capitalize() in fractal_protagonists_set or c_word in fractal_protagonists_set:
                            targets_in_context.add(c_word.capitalize())
                            
                    # 3. Aggregate score for all pairs found (undirected)
                    for char_B in targets_in_context:
                        if char_A != char_B:
                            # Use canonical ordering for undirected key (A, B) -> (min, max)
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

def generate_pyvis_network(G_fractal, centrality_scores, fractal_protagonists_names, kanda_name, graph_type, filename):
    """Generates an interactive Pyvis network plot."""
    
    is_directed = (graph_type == 'directed')
    
    # Initialize Pyvis Network
    net = Network(height="750px", width="100%", directed=is_directed, notebook=False, 
                  heading=f"{kanda_name} Fractal Protagonist Network ({graph_type.capitalize()} Sentiment)")
    
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

    for node in G_fractal.nodes():
        eigen_score = centrality_scores.get(node, 0)
        node_size = (eigen_score / max_eigen) * NODE_SIZE_SCALER + 10 # Scale size
        
        # Node Title for hover information
        node_title = f"<b>{node}</b><br>Prestige (Eigenvector): {eigen_score:.4f}"

        net.add_node(
            node, 
            label=node, 
            title=node_title, 
            value=node_size, # Size by Eigenvector Centrality (Prestige)
            color='#1E90FF', # Consistent Node Color (Dodger Blue)
            font={'size': 14, 'face': 'Arial', 'color': 'black'}
        )

    # 2. Add Edges
    max_weight = max([d['count'] for u, v, d in G_fractal.edges(data=True)]) if G_fractal.edges() else 1

    for u, v, data in G_fractal.edges(data=True):
        sentiment = data['sentiment']
        count = data['count']
        
        edge_color = get_sentiment_color(sentiment)
        edge_title = f"Avg. Sentiment: {sentiment:.3f} ({get_sentiment_title(sentiment)})<br>Interactions: {count}"
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

def plot_pair_sentiment_trajectory(kanda_names, all_sentiment_scores_by_pair):
    """Identifies pairs with major sentiment shifts and plots their trajectory."""
    
    print("\n--- Analyzing Relationship Trajectories (Matplotlib Line Plot) ---")
    
    pairs_to_plot = {} # Stores { (u, v): [score1, score2, ...] }
    
    for pair, scores in all_sentiment_scores_by_pair.items():
        if len(scores) >= 2: # Must be present in at least two Kandas
            score_values = [score for kanda, score in scores]
            sentiment_swing = max(score_values) - min(score_values)
            
            if sentiment_swing >= TRAJECTORY_THRESHOLD:
                pairs_to_plot[pair] = scores
                
    if not pairs_to_plot:
        print(f"No character pairs found with a significant sentiment swing (> {TRAJECTORY_THRESHOLD}).")
        return

    # --- FIX IMPLEMENTATION: Plotting in pages ---
    
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
            
            plt.title(f'{u} - {v}', fontsize=12) # Use hyphen for undirected relationship clarity
            plt.xticks(kanda_indices, kanda_labels, rotation=45, ha='right', fontsize=8)
            plt.ylabel('Avg. VADER Compound Score', fontsize=10)
            plt.ylim(-1.0, 1.0) # Consistent Y-axis range for VADER compound score
            plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        filename = os.path.join(OUTPUT_SENTIMENT_DIR, f'sentiment_relationship_trajectories_page_{page_num // PLOTS_PER_PAGE + 1}.png')
        plt.savefig(filename, dpi=200) # Reduced DPI slightly for faster rendering
        plt.close()
        print(f"✅ Trajectory plot page {page_num // PLOTS_PER_PAGE + 1} saved as: {filename}")


# --- Main Execution ---

def main():
    print("--- Starting Graph-Enhanced Sentiment Analysis (VADER/Pyvis) ---")
    try:
        text_data, character_set = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not os.path.exists(OUTPUT_SENTIMENT_DIR):
        os.makedirs(OUTPUT_SENTIMENT_DIR)

    # 1. TEMPORAL METRICS (Needed for Fractal Protagonist Identification - Directed Only)
    temporal_metrics_directed = []
    
    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda')
        if not kanda_name: continue 

        G_directed = build_directed_graph_for_kanda(chapter_data, character_set)
        metrics_directed = calculate_kanda_metrics(G_directed, kanda_name, 'directed')
        temporal_metrics_directed.append(metrics_directed)

    # 2. IDENTIFY FRACTAL PROTAGONISTS (Based on Directed Centrality)
    fractal_protagonists_names = calculate_fractal_protagonists_names(temporal_metrics_directed)
    fractal_protagonists_set = set(fractal_protagonists_names)
    print(f"\nIdentified {len(fractal_protagonists_names)} Fractal Protagonists.")
    
    # --- 3. CALCULATE SENTIMENT FOR DIRECTED (Conversational) NETWORK ---
    
    directed_sentiment_data = analyze_sentiment_interactions_kanda_wise(text_data, fractal_protagonists_set)
    
    # Aggregate data across all Kandas for the Overall/Master plot (Directed)
    directed_overall_raw_scores = defaultdict(list)
    kanda_names_list = []
    
    for kanda, edges in directed_sentiment_data.items():
        kanda_names_list.append(kanda)
        for (u, v), data in edges.items():
            directed_overall_raw_scores[(u, v)].append((kanda, data['sentiment'])) 

    # --- 4. CALCULATE SENTIMENT FOR UNDIRECTED (Co-occurrence) NETWORK ---
    
    undirected_sentiment_data = analyze_sentiment_interactions_undirected(text_data, fractal_protagonists_set)
    
    # Aggregate data across all Kandas for the Overall/Master plot (Undirected)
    undirected_overall_raw_scores = defaultdict(list)
    for kanda, edges in undirected_sentiment_data.items():
        for (u, v), data in edges.items():
            undirected_overall_raw_scores[(u, v)].append((kanda, data['sentiment'])) 

    # --- 5. DIRECTED PLOTS GENERATION (PYVIS) ---
    
    # Build Overall Directed Graph
    directed_overall_edges = {}
    for (u, v), scores in directed_overall_raw_scores.items():
        avg_score = np.mean([score for kanda, score in scores])
        total_count = sum([directed_sentiment_data[kanda][(u,v)]['count'] 
                           for kanda in directed_sentiment_data 
                           if (u,v) in directed_sentiment_data[kanda]])
                           
        directed_overall_edges[(u, v)] = {'count': total_count, 'sentiment': avg_score}
    
    G_directed_overall = nx.DiGraph()
    G_directed_overall.add_nodes_from(fractal_protagonists_names)
    for (u, v), data in directed_overall_edges.items():
        G_directed_overall.add_edge(u, v, **data)
        
    G_directed_filtered = G_directed_overall.subgraph([n for n in G_directed_overall.nodes() if G_directed_overall.degree(n) > 0]).copy()
    G_directed_undirected_temp = G_directed_filtered.to_undirected() 
    overall_eigenvector = nx.eigenvector_centrality(G_directed_undirected_temp, weight='weight', max_iter=1000)
    
    generate_pyvis_network(G_directed_filtered, overall_eigenvector, fractal_protagonists_names, "Overall", 'directed', os.path.join(OUTPUT_SENTIMENT_DIR, 'sentiment_overall_directed_network.html'))
    
    # Kanda-wise Directed Plots
    for kanda_name, edges in directed_sentiment_data.items():
        if not edges: continue
        G_kanda = nx.DiGraph()
        G_kanda.add_nodes_from(fractal_protagonists_names)
        for (u, v), data in edges.items():
             G_kanda.add_edge(u, v, **data)
        G_kanda_filtered = G_kanda.subgraph([n for n in G_kanda.nodes() if G_kanda.degree(n) > 0]).copy()
        G_kanda_undirected = G_kanda_filtered.to_undirected()
        kanda_eigenvector = nx.eigenvector_centrality(G_kanda_undirected, weight='weight', max_iter=1000)
        generate_pyvis_network(G_kanda_filtered, kanda_eigenvector, fractal_protagonists_names, kanda_name, 'directed', os.path.join(OUTPUT_SENTIMENT_DIR, f'sentiment_{kanda_name.replace(" ", "_")}_directed_network.html'))


    # --- 6. UNDIRECTED PLOTS GENERATION (PYVIS) ---

    # Build Overall Undirected Graph
    undirected_overall_edges = {}
    for (u, v), scores in undirected_overall_raw_scores.items():
        avg_score = np.mean([score for kanda, score in scores])
        total_count = sum([undirected_sentiment_data[kanda][(u,v)]['count'] 
                           for kanda in undirected_sentiment_data 
                           if (u,v) in undirected_sentiment_data[kanda]])
        undirected_overall_edges[(u, v)] = {'count': total_count, 'sentiment': avg_score}
    
    G_undirected_overall = nx.Graph()
    G_undirected_overall.add_nodes_from(fractal_protagonists_names)
    for (u, v), data in undirected_overall_edges.items():
        G_undirected_overall.add_edge(u, v, **data)
        
    G_undirected_filtered = G_undirected_overall.subgraph([n for n in G_undirected_overall.nodes() if G_undirected_overall.degree(n) > 0]).copy()
    overall_eigenvector_undirected = nx.eigenvector_centrality(G_undirected_filtered, weight='count', max_iter=1000)
    
    generate_pyvis_network(G_undirected_filtered, overall_eigenvector_undirected, fractal_protagonists_names, "Overall", 'undirected', os.path.join(OUTPUT_SENTIMENT_DIR, 'sentiment_overall_undirected_network.html'))

    # Kanda-wise Undirected Plots
    for kanda_name, edges in undirected_sentiment_data.items():
        if not edges: continue
        G_kanda = nx.Graph()
        G_kanda.add_nodes_from(fractal_protagonists_names)
        for (u, v), data in edges.items():
             G_kanda.add_edge(u, v, **data)
        G_kanda_filtered = G_kanda.subgraph([n for n in G_kanda.nodes() if G_kanda.degree(n) > 0]).copy()
        kanda_eigenvector = nx.eigenvector_centrality(G_kanda_filtered, weight='count', max_iter=1000)
        generate_pyvis_network(G_kanda_filtered, kanda_eigenvector, fractal_protagonists_names, kanda_name, 'undirected', os.path.join(OUTPUT_SENTIMENT_DIR, f'sentiment_{kanda_name.replace(" ", "_")}_undirected_network.html'))


    # --- 7. RELATIONSHIP TRAJECTORY PLOT (MATPLOTLIB) ---
    
    # We plot the sentiment trajectory for the Directed Network as it best represents relationships
    plot_pair_sentiment_trajectory(kanda_names_list, directed_overall_raw_scores)

    print("\n--- Final Sentiment Analysis Summary ---")
    print(f"All sentiment networks (Overall + 7 Kandas) are saved as interactive HTML files in the '{OUTPUT_SENTIMENT_DIR}' directory.")
    print(f"The relationship trajectory plot (line chart) is saved as a PNG.")


if __name__ == '__main__':
    main()