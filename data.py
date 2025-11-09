import re
import json
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import itertools

with open("dataset/ramayan_trunc.txt", "r", encoding="utf-8") as f:
    text = f.read()

def extract_books_and_cantos(text, num_books=None, cantos_per_book=None):
    book_pattern = re.compile(
        r'BOOK\s+([IVXLC]+)\.?\s*(.*?)(?=BOOK\s+[IVXLC]+\.?|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    book_matches = book_pattern.findall(text)
    print(f"Found {len(book_matches)} books in the text")

    if num_books:
        book_matches = book_matches[:num_books]
        print(f"Analyzing first {num_books} books")

    all_cantos = []
    def roman_to_int(roman):
        vals = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100}
        total, prev = 0, 0
        for ch in reversed(roman.upper()):
            v = vals.get(ch, 0)
            total += -v if v < prev else v
            prev = v
        return total

    for book_roman, book_text in book_matches:
        book_num = roman_to_int(book_roman)
        canto_pattern = re.compile(
            r'Canto\s+([IVXLC]+)\.\s*(.*?)(?=Canto\s+[IVXLC]+\.|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        canto_matches = canto_pattern.findall(book_text)
        print(f"  Book {book_roman} ({book_num}): Found {len(canto_matches)} cantos")

        if cantos_per_book:
            canto_matches = canto_matches[:cantos_per_book]

        for canto_roman, canto_text in canto_matches:
            canto_num = roman_to_int(canto_roman)
            canto_text_clean = canto_text.replace('\n', ' ')
            all_cantos.append((book_num, canto_num, canto_text_clean))
    return all_cantos

NUM_BOOKS_TO_ANALYZE = None
NUM_CANTOS_PER_BOOK = None
cantos = extract_books_and_cantos(text, NUM_BOOKS_TO_ANALYZE, NUM_CANTOS_PER_BOOK)

with open("dataset/principal_names_list.json", "r", encoding="utf-8") as f:
    principal_names = json.load(f)

full_text = " ".join(canto[2] for canto in cantos)
freqs = {name: full_text.count(name) for name in principal_names}
top10 = sorted(freqs.items(), key=lambda x: x[1], reverse=True)[:10]
print("\nTop 10 Mentions:")
for n, c in top10:
    print(f"{n}: {c}")

wordcloud = WordCloud(width=1000, height=600, background_color='white').generate_from_frequencies(freqs)
plt.figure(figsize=(12,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Principal Names Word Cloud", fontsize=16)
plt.show()

window_size = 300  # adjust for context window
cooccur = Counter()

for canto in cantos:
    text_segment = canto[2]
    for name1 in principal_names:
        if name1 in text_segment:
            for name2 in principal_names:
                if name1 != name2 and name2 in text_segment:
                    pair = tuple(sorted((name1, name2)))
                    cooccur[pair] += 1

# top pairs
top_pairs = cooccur.most_common(10)
print("\nTop 10 Co-occurring Pairs:")
for (a, b), c in top_pairs:
    print(f"{a} - {b}: {c}")


# === STEP 1: Compute pair counts across cantos ===
pair_counts = Counter()

for book_num, canto_num, canto_text in cantos: 
    present = []
    for char in principal_names:
        pattern = char
        pattern = pattern.replace('a', '[aá]').replace('i', '[ií]').replace('u', '[uú]')
        pattern = pattern.replace('n', '[nṇ]').replace('r', '[rṛ]').replace('s', '[sś]')
        
        if re.search(rf"\b{pattern}\b", canto_text, re.IGNORECASE):
            present.append(char)
    
    for a, b in itertools.combinations(sorted(set(present)), 2):
        pair_counts[(a, b)] += 1

node_weight = defaultdict(int)
for (a, b), w in pair_counts.items():
    node_weight[a] += w
    node_weight[b] += w

# Only keep nodes with frequency > 5
valid_nodes = {n for n, w in node_weight.items()}
print(len(valid_nodes))

# Build graph
G = nx.Graph()
for (a, b), w in pair_counts.items():
    if a in valid_nodes and b in valid_nodes and w >= 2:
        G.add_edge(a, b, weight=w)

# Node sizes
sizes = [node_weight[n]*0.1 for n in G.nodes()]

# === STEP 3: Visualize ===
plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, k=1.2, seed=42, iterations=60)

edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
nx.draw_networkx_edges(G, pos, edgelist=edges, width=[w*0.2 for w in weights], alpha=0.4)
nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="lightblue", edgecolors="black", alpha=0.8)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")

plt.title("Character Co-occurrence Network — Ramayana (First 20 Cantos, freq>5)", fontsize=16)
plt.axis("off")
plt.tight_layout()
plt.show()

import networkx as nx

# Basic network stats
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
avg_degree = sum(dict(G.degree()).values()) / num_nodes
density = nx.density(G)

# Diameter (only for connected component)
if nx.is_connected(G):
    diameter = nx.diameter(G)
else:
    # Take the largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_lcc = G.subgraph(largest_cc)
    diameter = nx.diameter(G_lcc)

# Average clustering coefficient
avg_clustering = nx.average_clustering(G, weight='weight')

# Degree assortativity
degree_assortativity = nx.degree_assortativity_coefficient(G)

# Top 5 nodes by degree
top_nodes = sorted(G.degree(weight='weight'), key=lambda x: x[1], reverse=True)[:5]

# Print results
print(f"Total nodes: {num_nodes}")
print(f"Total edges: {num_edges}")
print(f"Average degree: {avg_degree:.2f}")
print(f"Density: {density:.3f}")
print(f"Diameter: {diameter}")
print(f"Average clustering coefficient: {avg_clustering:.3f}")
print(f"Degree assortativity: {degree_assortativity:.3f}")
print("Top 5 nodes by weighted degree:", top_nodes)
