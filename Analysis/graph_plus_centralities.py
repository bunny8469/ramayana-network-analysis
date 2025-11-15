import json
import re
import networkx as nx
from collections import Counter

# --- Configuration ---
FILE_TEXT = '/Users/chetanvellanki/Desktop/ramayana-network-analysis/data/ramayana_resolved_final.json'
FILE_CANONICAL_NAMES = '/Users/chetanvellanki/Desktop/ramayana-network-analysis/data/canonical_characters.json'
CO_OCCURRENCE_WINDOW = 15 # Max words distance for an undirected edge
OUTPUT_GRAPH_DATA = 'ramayana_graphs_and_centrality.json'

# Discourse markers for inferring speech direction (Source -> Target)
# E.g., "Rama said to Sita" -> Rama is Source, Sita is Target
DISCOURSE_MARKERS = [
    r'\b(said|spoke|told|asked|replied|enquired|declared|commanded|adressed|answered|responded|cried|whispered|suggested|stated)\b',
]
DISCOURSE_MARKER_PATTERN =  re.compile('|'.join(DISCOURSE_MARKERS), re.IGNORECASE)

# --- 1. Data Loading and Preparation ---

def load_data():
    """Loads all necessary JSON files."""
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

    # Convert the list of canonical names into a set for O(1) lookups
    character_set = set(canonical_names)
    return text_data, character_set

def tokenize_and_tag(canto_text, character_set):
    """Tokenizes text, replaces newlines/excess spaces, and tags words as characters/non-characters."""
    # Simple tokenizer that splits on non-word characters and removes empty strings
    words = re.findall(r'\b\w+\b', canto_text.strip())

    # Create a list of tuples: [(word, is_character, canonical_name)]
    tagged_tokens = []
    for word in words:
        # Check against the canonical set (case-sensitive as names were resolved to canonical casing)
        is_char = word in character_set
        tagged_tokens.append((word, is_char))
    
    return tagged_tokens

# --- 2. Graph Construction ---

def build_undirected_graph(text_data, character_set):
    """Builds the Undirected (Co-occurrence) Graph."""
    G_undirected = nx.Graph()
    G_undirected.add_nodes_from(character_set)
    co_occurrence_edges = Counter()

    print("Building Undirected Graph (Co-occurrence)...")
    for chapter in text_data['chapters']:
        for canto_text in chapter['cantos']:
            tagged_tokens = tokenize_and_tag(canto_text, character_set)
            
            # Find all character positions in the canto
            char_positions = [(i, token[0]) for i, token in enumerate(tagged_tokens) if token[1]]
            
            # Use a sliding window to find co-occurrences
            for i in range(len(char_positions)):
                char_A = char_positions[i][1]
                pos_A = char_positions[i][0]
                
                # Check characters appearing up to the CO_OCCURRENCE_WINDOW ahead
                for j in range(i + 1, len(char_positions)):
                    char_B = char_positions[j][1]
                    pos_B = char_positions[j][0]

                    # Stop if the distance exceeds the window size
                    if pos_B - pos_A > CO_OCCURRENCE_WINDOW:
                        break 
                    
                    if char_A != char_B:
                        # Use a sorted tuple to ensure edges are counted only once (A, B) not (B, A)
                        edge = tuple(sorted((char_A, char_B)))
                        co_occurrence_edges[edge] += 1

    # Add edges to the networkx graph
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
            
            # Scan for a sequence: [Char_A] [Discourse Marker] [to/towards/with] [Char_B]
            for i, (word, is_char) in enumerate(tagged_tokens):
                if is_char:
                    char_A = word # Potential speaker (Source)
                    
                    # Look ahead for a discourse marker and another character (Target)
                    # Window is limited to CO_OCCURRENCE_WINDOW for efficiency and context relevance
                    for j in range(i + 1, min(i + CO_OCCURRENCE_WINDOW, len(tagged_tokens))):
                        # Check for discourse marker (said/told/asked etc.)
                        if DISCOURSE_MARKER_PATTERN.search(tagged_tokens[j][0]):
                            
                            # Now look ahead for the conversational target (Target)
                            # Look up to 5 words further for a target character
                            for k in range(j + 1, min(j + 5, len(tagged_tokens))):
                                if tagged_tokens[k][1] and tagged_tokens[k][0] != char_A:
                                    char_B = tagged_tokens[k][0] # Conversational Target
                                    
                                    # Found a directed conversation: Char_A (Source) -> Char_B (Target)
                                    conversational_edges[(char_A, char_B)] += 1
                                    # Break the inner loop once a target is found for this speaker
                                    break 

    # Add edges to the networkx graph
    for (u, v), weight in conversational_edges.items():
        G_directed.add_edge(u, v, weight=weight)
        
    return G_directed, conversational_edges

# --- 3. Centrality Calculation ---

def calculate_centrality(G, graph_type):
    """Calculates Degree and Eigenvector centrality for a graph."""
    print(f"Calculating Centrality for {graph_type} Graph...")
    
    # Calculate Degree Centrality
    if graph_type == 'Directed':
        # Use In-Degree for directed graph (how many times a character is spoken TO)
        degree_centrality = nx.in_degree_centrality(G)
    else:
        degree_centrality = nx.degree_centrality(G)

    # Calculate Eigenvector Centrality
    try:
        # Eigenvector centrality is slightly more complex and can fail on certain graphs
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000)
    except nx.NetworkXNoConvergence:
        print(f"Warning: Eigenvector centrality failed to converge for {graph_type}. Using unweighted calculation.")
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    except Exception as e:
        print(f"Error during Eigenvector centrality calculation for {graph_type}: {e}")
        eigenvector_centrality = {node: 0.0 for node in G.nodes()}

    # Compile the results
    centrality_results = {}
    for node in G.nodes():
        centrality_results[node] = {
            'degree_centrality': degree_centrality.get(node, 0.0),
            'eigenvector_centrality': eigenvector_centrality.get(node, 0.0),
        }
        
    return centrality_results

# --- Main Execution ---

def main():
    try:
        text_data, character_set = load_data()
    except Exception:
        return

    # Filter out characters that never appear in the text
    # This cleans the graphs for analysis
    all_appearing_chars = set()
    for chapter in text_data['chapters']:
        for canto_text in chapter['cantos']:
            for word in re.findall(r'\b\w+\b', canto_text):
                if word in character_set:
                    all_appearing_chars.add(word)
    
    # Re-filter the node set to only include characters that appear
    character_set_filtered = all_appearing_chars

    # 1. Undirected Graph
    G_undirected, co_occurrence_edges = build_undirected_graph(text_data, character_set_filtered)
    
    # 2. Directed Graph
    G_directed, conversational_edges = build_directed_graph(text_data, character_set_filtered)
    
    # Filter out nodes in the graphs that have no connections (degree 0)
    G_undirected_filtered = G_undirected.subgraph([n for n, d in G_undirected.degree() if d > 0]).copy()
    G_directed_filtered = G_directed.subgraph([n for n in G_directed.nodes() if G_directed.in_degree(n) > 0 or G_directed.out_degree(n) > 0]).copy()


    # 3. Centrality Measures
    undirected_centrality = calculate_centrality(G_undirected_filtered, 'Undirected')
    directed_centrality = calculate_centrality(G_directed_filtered, 'Directed')

    # Prepare final output structure
    results = {
        "metadata": {
            "total_canonical_characters": len(character_set),
            "total_appearing_characters": len(character_set_filtered),
            "undirected_graph": {
                "nodes": len(G_undirected_filtered.nodes()),
                "edges": len(G_undirected_filtered.edges()),
                "description": f"Co-occurrence network (window={CO_OCCURRENCE_WINDOW} words)."
            },
            "directed_graph": {
                "nodes": len(G_directed_filtered.nodes()),
                "edges": len(G_directed_filtered.edges()),
                "description": "Conversational network (Source -> Target using discourse markers)."
            }
        },
        "centrality_results_undirected": undirected_centrality,
        "centrality_results_directed": directed_centrality,
        # Optionally include edge lists for visualization later
        "undirected_edges_top_10": sorted(co_occurrence_edges.items(), key=lambda item: item[1], reverse=True)[:10],
        "directed_edges_top_10": sorted(conversational_edges.items(), key=lambda item: item[1], reverse=True)[:10]
    }

    # Save the final analysis to a JSON file
    with open(OUTPUT_GRAPH_DATA, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n✅ Network analysis complete. Results saved to: {OUTPUT_GRAPH_DATA}")
    print("\n--- Summary ---")
    print(f"Undirected Graph: Nodes={results['metadata']['undirected_graph']['nodes']}, Edges={results['metadata']['undirected_graph']['edges']}")
    print(f"Directed Graph: Nodes={results['metadata']['directed_graph']['nodes']}, Edges={results['metadata']['directed_graph']['edges']}")


if __name__ == '__main__':
    main()