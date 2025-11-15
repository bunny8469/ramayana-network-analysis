import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os 
import numpy as np
import itertools # For generating character pairs

# --- Configuration (Based on previous analysis) ---
FILE_TEXT = './../data/ramayana_resolved_final.json'
FILE_CANONICAL_NAMES = './../data/canonical_characters.json'
CO_OCCURRENCE_WINDOW = 15 
TOP_N_FRACTAL_PROTAGONISTS = 30 # Characters considered in the sentiment analysis

# --- Sentiment Configuration ---
# Custom, Rule-Based Lexicon (Higher magnitude = stronger sentiment)
SENTIMENT_LEXICON = {
    'joy': 2.5, 'happy': 2.0, 'delight': 2.0, 'bless': 1.5, 'righteous': 1.5, 'virtuous': 1.0,
    'great': 1.0, 'mighty': 1.0, 'brave': 1.0, 'love': 2.5, 'truth': 1.5,
    'anger': -3.0, 'wrath': -3.0, 'fury': -2.5, 'hate': -2.5, 'destroy': -2.0, 'kill': -2.0, 
    'sorrow': -2.0, 'grief': -2.0, 'weep': -1.0, 'afraid': -1.0, 'fear': -1.0, 'sin': -1.5, 
    'treachery': -2.5, 'evil': -2.5, 'demon': -1.0, 'rakshasa': -1.0
}
SENTIMENT_THRESHOLD = 0.5 # Threshold to define positive/negative vs. neutral

# Visualization & Output Configuration
NODE_SIZE_SCALER = 15000 
EDGE_WIDTH_SCALER = 4 
OUTPUT_SENTIMENT_DIR = 'sentiment_plots/'
TRAJECTORY_THRESHOLD = 1.0 # Minimum swing in sentiment (e.g., from -1.0 to +0.5)

# Discourse markers for conversational graph (Source -> Target)
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- Core Utility Functions ---

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

def calculate_kanda_metrics(G, kanda_name):
    """Calculates centrality for a Kanda-specific graph (required for fractal scoring)."""
    metrics = {'kanda': kanda_name, 'type': 'directed', 'centrality_scores': {}, 'active_nodes_count': len(G.nodes())}
    if len(G.nodes()) > 1:
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
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

# --- 1. Sentiment Scoring Logic ---

def score_conversation_text(text):
    """Scores a text snippet based on the sentiment lexicon."""
    total_score = 0
    word_count = 0
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    for word in clean_text.split():
        score = SENTIMENT_LEXICON.get(word, 0)
        total_score += score
        if score != 0:
            word_count += 1
    
    return total_score / word_count if word_count > 0 else 0

def analyze_sentiment_interactions_kanda_wise(text_data, fractal_protagonists_set):
    """Scans text and returns raw scores structured by Kanda."""
    
    all_kanda_raw_scores = {}
    
    for chapter in text_data['chapters']:
        kanda_name = chapter.get('kanda')
        if not kanda_name: continue
        
        raw_sentiment_scores = defaultdict(list)
        canto_texts = chapter.get('cantos', [])
        
        for canto_text in canto_texts:
            sentences = re.split(r'(?<=[.?!])\s+', canto_text)
            
            for sentence in sentences:
                words = re.findall(r'\b\w+\b', sentence)
                fractal_chars_in_sentence = [word for word in words if word in fractal_protagonists_set]
                
                if len(fractal_chars_in_sentence) < 2: continue

                # Simple inference of directed conversation
                for char_A in fractal_chars_in_sentence:
                    if any(DISCOURSE_MARKER_PATTERN.search(word) for word in words):
                        for char_B in fractal_chars_in_sentence:
                            if char_A != char_B:
                                score = score_conversation_text(sentence)
                                if score != 0:
                                    raw_sentiment_scores[(char_A, char_B)].append(score)
                                    break # Only capture the first inferred interaction per sentence

        # Aggregate raw scores into final average sentiment for this Kanda
        kanda_final_edges = {}
        for (u, v), scores in raw_sentiment_scores.items():
            avg_score = np.mean(scores)
            # IMPORTANT: The edge data must include 'count' for later aggregation
            kanda_final_edges[(u, v)] = {'count': len(scores), 'sentiment': avg_score}
            
        all_kanda_raw_scores[kanda_name] = kanda_final_edges
        
    return all_kanda_raw_scores

# --- 2. Kanda-Wise Visualization ---

def plot_sentiment_network(G_directed_fractal, centrality_scores, fractal_protagonists_names, kanda_name, filename):
    """Generates the plot of fractal protagonists colored by sentiment."""
    
    plt.figure(figsize=(15, 12))
    
    if kanda_name == "Overall":
        title = 'Overall Fractal Protagonists Conversation Network (Sentiment)'
    else:
        title = f'Kanda-wise Sentiment: {kanda_name}'
        
    plt.title(title, fontsize=18)
    
    # Use spring layout based on the undirected projection
    pos = nx.spring_layout(G_directed_fractal.to_undirected(), k=0.1, iterations=50, seed=42)

    # 1. Prepare Node Sizes (based on centrality in the plotted graph)
    max_eigen = max(centrality_scores.values()) if centrality_scores else 1
    node_sizes = [centrality_scores.get(node, 0) / max_eigen * NODE_SIZE_SCALER for node in G_directed_fractal.nodes()]
    
    # 2. Prepare Edges (Color and Width)
    edge_colors = []
    edge_widths = []
    
    # Find max weight for scaling edge width
    max_weight = max([d['count'] for u, v, d in G_directed_fractal.edges(data=True)]) if G_directed_fractal.edges() else 1

    for u, v, data in G_directed_fractal.edges(data=True):
        sentiment = data['sentiment']
        count = data['count']
        
        # Color based on sentiment threshold
        if sentiment > SENTIMENT_THRESHOLD:
            edge_colors.append('green') # Positive
        elif sentiment < -SENTIMENT_THRESHOLD:
            edge_colors.append('red') # Negative
        else:
            edge_colors.append('gray') # Neutral
            
        # Width based on conversation volume (count)
        edge_widths.append((count / max_weight) * EDGE_WIDTH_SCALER + 0.5)

    # Draw Nodes
    nx.draw_networkx_nodes(
        G_directed_fractal, pos, 
        node_size=node_sizes, 
        node_color='skyblue', 
        alpha=0.9, 
        linewidths=1.5, 
        edgecolors='black'
    )
    
    # Draw Edges
    nx.draw_networkx_edges(
        G_directed_fractal, pos, 
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=True,
        arrowsize=15
    )
    
    # Draw Labels (Bold, clear names)
    labels = {name: name for name in fractal_protagonists_names if name in G_directed_fractal.nodes()}
    nx.draw_networkx_labels(G_directed_fractal, pos, labels, font_size=10, font_weight='bold')
    
    # Add custom legend for sentiment color
    legend_elements = [
        plt.Line2D([0], [0], color='green', lw=4, label='Positive Conversation'),
        plt.Line2D([0], [0], color='red', lw=4, label='Negative Conversation'),
        plt.Line2D([0], [0], color='gray', lw=4, label='Neutral Conversation')
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=12)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Sentiment Network plot saved as: {filename}")


# --- 3. Relationship Trajectory Plot ---

def plot_pair_sentiment_trajectory(kanda_names, all_sentiment_scores_by_pair):
    """Identifies pairs with major sentiment shifts and plots their trajectory."""
    
    print("\n--- Analyzing Relationship Trajectories ---")
    
    pairs_to_plot = {} # Stores { (u, v): [score1, score2, ...] }
    
    for pair, scores in all_sentiment_scores_by_pair.items():
        if len(scores) >= 2: # Must be present in at least two Kandas
            score_values = [score for kanda, score in scores]
            
            # Find max sentiment swing (Difference between max and min score)
            sentiment_swing = max(score_values) - min(score_values)
            
            # Check for significant shift (positive to negative or vice versa)
            if sentiment_swing >= TRAJECTORY_THRESHOLD:
                pairs_to_plot[pair] = scores
                
    if not pairs_to_plot:
        print("No character pairs found with a significant sentiment swing (> 1.0).")
        return

    num_plots = len(pairs_to_plot)
    cols = 2
    rows = (num_plots + cols - 1) // cols
    
    plt.figure(figsize=(8 * cols, 5 * rows))
    
    print(f"Visualizing {len(pairs_to_plot)} pairs with major sentiment swings:")

    for i, ((u, v), scores) in enumerate(pairs_to_plot.items()):
        plt.subplot(rows, cols, i + 1)
        
        # Ensure kanda names are aligned with scores
        kanda_indices = [kanda_names.index(kanda) for kanda, score in scores]
        kanda_labels = [kanda for kanda, score in scores]
        score_values = [score for kanda, score in scores]
        
        # Plotting the trajectory
        plt.plot(kanda_indices, score_values, marker='o', linestyle='-', linewidth=2, alpha=0.7, color='blue')
        
        # Highlight start and end sentiment
        start_score = score_values[0]
        end_score = score_values[-1]

        plt.plot(kanda_indices[0], start_score, marker='o', color='green' if start_score > 0 else 'red', markersize=8)
        plt.plot(kanda_indices[-1], end_score, marker='o', color='green' if end_score > 0 else 'red', markersize=8, label='Final Score')
        
        # Draw the neutral/threshold line
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        
        plt.title(f'{u} → {v}', fontsize=14)
        plt.xticks(kanda_indices, kanda_labels, rotation=45, ha='right', fontsize=10)
        plt.ylabel('Avg. Sentiment Score', fontsize=12)
        plt.ylim(-3.5, 3.5) # Consistent Y-axis range
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    filename = os.path.join(OUTPUT_SENTIMENT_DIR, 'sentiment_relationship_trajectories.png')
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"✅ Relationship Trajectory plot saved as: {filename}")
    print(f"Summary: The plot shows how the conversational tone between key pairs shifted over the Kandas.")


# --- Main Execution ---

def main():
    print("--- Starting Graph-Enhanced Sentiment Analysis ---")
    try:
        text_data, character_set = load_data()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    if not os.path.exists(OUTPUT_SENTIMENT_DIR):
        os.makedirs(OUTPUT_SENTIMENT_DIR)

    # 1. TEMPORAL METRICS (Needed for Fractal Protagonist Identification)
    temporal_metrics_directed = []
    
    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda')
        if not kanda_name: continue 

        G_directed = build_directed_graph_for_kanda(chapter_data, character_set)
        metrics_directed = calculate_kanda_metrics(G_directed, kanda_name)
        temporal_metrics_directed.append(metrics_directed)

    # 2. IDENTIFY FRACTAL PROTAGONISTS
    fractal_protagonists_names = calculate_fractal_protagonists_names(temporal_metrics_directed)
    fractal_protagonists_set = set(fractal_protagonists_names)
    print(f"\nIdentified {len(fractal_protagonists_names)} Fractal Protagonists.")
    
    # --- 3. CALCULATE SENTIMENT FOR ALL KANDAS AND AGGREGATE ---
    
    kanda_sentiment_data = analyze_sentiment_interactions_kanda_wise(text_data, fractal_protagonists_set)
    
    # Aggregate data across all Kandas for the Overall/Master plot
    overall_raw_scores = defaultdict(list)
    kanda_names_list = []
    
    for kanda, edges in kanda_sentiment_data.items():
        kanda_names_list.append(kanda)
        for (u, v), data in edges.items():
            # Store sentiment for trajectory plotting
            overall_raw_scores[(u, v)].append((kanda, data['sentiment'])) 

    # --- 4. OVERALL PLOT GENERATION ---
    
    overall_edges = {}
    for (u, v), scores in overall_raw_scores.items():
        avg_score = np.mean([score for kanda, score in scores])
        
        # FIX: Correctly iterate through Kandas and access the specific edge's 'count'
        total_count = sum([kanda_sentiment_data[kanda][(u,v)]['count'] 
                           for kanda in kanda_sentiment_data 
                           if (u,v) in kanda_sentiment_data[kanda]])
                           
        overall_edges[(u, v)] = {'count': total_count, 'sentiment': avg_score}
    
    # Build Overall Graph
    G_overall = nx.DiGraph()
    G_overall.add_nodes_from(fractal_protagonists_names)
    for (u, v), data in overall_edges.items():
        G_overall.add_edge(u, v, **data)
        
    G_overall_filtered = G_overall.subgraph([n for n in G_overall.nodes() if G_overall.degree(n) > 0]).copy()
    G_undirected_temp = G_overall_filtered.to_undirected() 
    overall_eigenvector = nx.eigenvector_centrality(G_undirected_temp, weight='weight', max_iter=1000)
    
    plot_sentiment_network(G_overall_filtered, overall_eigenvector, fractal_protagonists_names, "Overall", os.path.join(OUTPUT_SENTIMENT_DIR, 'sentiment_overall_network.png'))
    
    # --- 5. KANDA-WISE PLOTS GENERATION ---
    
    print("\n--- Generating Kanda-wise Sentiment Networks ---")
    for kanda_name, edges in kanda_sentiment_data.items():
        if not edges: continue

        # Build Kanda-specific Graph (only includes active fractal protagonists)
        G_kanda = nx.DiGraph()
        G_kanda.add_nodes_from(fractal_protagonists_names)
        
        for (u, v), data in edges.items():
             G_kanda.add_edge(u, v, **data)
        
        G_kanda_filtered = G_kanda.subgraph([n for n in G_kanda.nodes() if G_kanda.degree(n) > 0]).copy()
        
        # Calculate centrality on the Kanda graph for dynamic node sizing
        G_kanda_undirected = G_kanda_filtered.to_undirected()
        kanda_eigenvector = nx.eigenvector_centrality(G_kanda_undirected, weight='weight', max_iter=1000)
        
        plot_sentiment_network(G_kanda_filtered, kanda_eigenvector, fractal_protagonists_names, kanda_name, os.path.join(OUTPUT_SENTIMENT_DIR, f'sentiment_{kanda_name.replace(" ", "_")}_network.png'))

    # --- 6. RELATIONSHIP TRAJECTORY PLOT ---
    
    plot_pair_sentiment_trajectory(kanda_names_list, overall_raw_scores)

    print("\n--- Final Sentiment Analysis Summary ---")
    print(f"All sentiment networks (Overall + 7 Kandas) and the relationship trajectory plot are saved in the '{OUTPUT_SENTIMENT_DIR}' directory.")


if __name__ == '__main__':
    main()