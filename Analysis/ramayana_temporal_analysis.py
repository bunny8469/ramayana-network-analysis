import json
import re
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

# --- Configuration ---
FILE_TEXT = './../data/ramayana_resolved_final.json'
FILE_CANONICAL_NAMES = './../data/canonical_characters.json'
CO_OCCURRENCE_WINDOW = 15 # Max words distance for an undirected edge
OUTPUT_ANALYSIS_DATA = 'ramayana_temporal_analysis.json'
PROTAGONISTS_TO_TRACK = ["Rama", "Sita", "Hanuman", "Lakshmana", "Ravana", "Kausalya", "Dasharatha"] # Added more key figures for tracking
TOP_N_FRACTAL_PROTAGONISTS = 30 # Number of top protagonists to identify and plot

# Discourse markers for conversational graph (Source -> Target)
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN = re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- Utility Functions ---

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

# --- Graph Building Functions ---

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

# --- Temporal Metrics Function ---

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
            print(f"Warning: Eigenvector centrality failed to converge for {kanda_name} ({graph_type}). Using unweighted calculation.")
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000) # Fallback to unweighted
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

        # Track protagonist scores
        for protag in PROTAGONISTS_TO_TRACK:
            metrics[f'{protag}_centrality'] = eigenvector_centrality.get(protag, 0.0)
            
    else:
        for protag in PROTAGONISTS_TO_TRACK:
            metrics[f'{protag}_centrality'] = 0.0

    return metrics

# --- Fractal Protagonist Analysis ---

def calculate_fractal_protagonists(temporal_data, graph_type='undirected'):
    """Calculates the fractal protagonist score based on average normalized rank across Kandas."""
    
    # Filter data for the chosen graph type
    filtered_data = [d for d in temporal_data if d['type'] == graph_type]
    
    # Store normalized ranks for every character across all Kandas
    character_ranks = defaultdict(list)
    
    for kanda_metrics in filtered_data:
        kanda_name = kanda_metrics['kanda']
        scores = kanda_metrics['centrality_scores']
        active_nodes_count = kanda_metrics['active_nodes_count']
        
        if active_nodes_count <= 1:
            # Skip Kandas with too few nodes for meaningful ranking
            continue
            
        # 1. Rank characters by Eigenvector Centrality
        ranked_chars = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        
        # Create a dictionary of {character: rank_number (starting at 1)}
        char_to_rank = {char: rank + 1 for rank, (char, score) in enumerate(ranked_chars)}
        
        # 2. Calculate Normalized Score for each character
        for char, rank in char_to_rank.items():
            # Normalized Score: (Total Characters - Rank) / Total Characters. Max score is close to 1.
            normalized_score = (active_nodes_count - rank) / active_nodes_count
            character_ranks[char].append(normalized_score)

    # 3. Calculate Fractal Score (Average Normalized Score)
    fractal_scores = {}
    for char, scores in character_ranks.items():
        # Only consider characters who appear in at least 50% of the Kandas
        if len(scores) >= len(filtered_data) / 2: 
            fractal_scores[char] = sum(scores) / len(scores)

    # 4. Get top N fractal protagonists
    top_fractal = sorted(
        fractal_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )[:TOP_N_FRACTAL_PROTAGONISTS]
    
    return top_fractal

def plot_fractal_protagonists(fractal_scores, graph_type):
    """Generates a horizontal bar chart of the top fractal protagonists."""
    
    characters = [item[0] for item in fractal_scores]
    scores = [item[1] for item in fractal_scores]
    
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

# --- TEMPORAL EVOLUTION PLOT FUNCTION (MISSING FUNCTION) ---
def plot_temporal_evolution(temporal_data):
    """Generates a line plot showing protagonist centrality evolution across Kandas."""
    
    # Extract the kanda names and filter data for undirected graph for plotting base
    kanda_sequence = [d['kanda'] for d in temporal_data if d['type'] == 'undirected']
    
    plt.figure(figsize=(15, 8))
    
    for protag in PROTAGONISTS_TO_TRACK:
        # Extract centrality scores for the current protagonist
        scores = [d[f'{protag}_centrality'] for d in temporal_data if d['type'] == 'undirected']
        
        plt.plot(kanda_sequence, scores, marker='o', linestyle='-', label=protag, linewidth=3)

    plt.title('Evolution of Character Centrality (Undirected Co-occurrence Network)', fontsize=16)
    plt.xlabel('Ramayana Kanda (Time Step)', fontsize=14)
    plt.ylabel('Eigenvector Centrality Score', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Protagonist', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = 'temporal_centrality_evolution.png'
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n✅ Temporal evolution plot saved as: {filename}")


# --- Main Execution ---

def main():
    print("--- Starting Ramayana Temporal Network Analysis ---")
    try:
        text_data, character_set = load_data()
    except Exception:
        return

    temporal_metrics = []

    # --- 1. KANDA-WISE METRICS COLLECTION ---
    for chapter_data in text_data['chapters']:
        kanda_name = chapter_data.get('kanda', 'Unknown Kanda')
        print(f"\nProcessing Kanda: {kanda_name}...")
        
        # Undirected Graph (Co-occurrence)
        G_undirected = build_graph_for_kanda(chapter_data, character_set, 'undirected')
        metrics_undirected = calculate_kanda_metrics(G_undirected, 'undirected', kanda_name)
        temporal_metrics.append(metrics_undirected)

        # Directed Graph (Conversational)
        G_directed = build_graph_for_kanda(chapter_data, character_set, 'directed')
        metrics_directed = calculate_kanda_metrics(G_directed, 'directed', kanda_name)
        temporal_metrics.append(metrics_directed)

    # 2. Save all temporal metrics to JSON
    with open(OUTPUT_ANALYSIS_DATA, 'w') as f:
        json.dump(temporal_metrics, f, indent=4)
        
    print(f"\n✅ Temporal metrics saved to: {OUTPUT_ANALYSIS_DATA}")
    
    # --- 3. FRACTAL PROTAGONIST ANALYSIS ---
    
    # Calculate and plot for Undirected Network
    undirected_fractal_protagonists = calculate_fractal_protagonists(temporal_metrics, 'undirected')
    print(f"\n--- Top {TOP_N_FRACTAL_PROTAGONISTS} Fractal Protagonists (Undirected) ---")
    for rank, (char, score) in enumerate(undirected_fractal_protagonists):
        print(f"Rank {rank+1}: {char} (Score: {score:.4f})")
    plot_fractal_protagonists(undirected_fractal_protagonists, 'undirected')
    
    # Calculate and plot for Directed Network
    directed_fractal_protagonists = calculate_fractal_protagonists(temporal_metrics, 'directed')
    print(f"\n--- Top {TOP_N_FRACTAL_PROTAGONISTS} Fractal Protagonists (Directed) ---")
    for rank, (char, score) in enumerate(directed_fractal_protagonists):
        print(f"Rank {rank+1}: {char} (Score: {score:.4f})")
    plot_fractal_protagonists(directed_fractal_protagonists, 'directed')


    # --- 4. TEMPORAL EVOLUTION PLOT ---
    # This call caused the NameError, now resolved by placing the definition above.
    plot_temporal_evolution(temporal_metrics)
    
    print("\n--- Temporal Analysis Complete ---")


if __name__ == '__main__':
    main()