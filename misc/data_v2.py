import re
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

with open("ramayan_trunc.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('\n', ' ')

canto_pattern = re.compile(
    r'(Canto\s+[IVXLC]+\..*?)(?=Canto\s+[IVXLC]+\.|\Z)',
    re.DOTALL
)
canto_matches = canto_pattern.findall(text)

cantos = canto_matches[:10]
print(f"Found {len(cantos)} cantos\n")

try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def normalize_name(name):
    # Remove possessive 's
    name = re.sub(r"'s$", "", name)
    
    # Remove special diacritics variants (normalize to base form)
    # Ráma, Rama -> Rama
    # name = name.replace('á', 'a').replace('í', 'i').replace('ú', 'u')
    # name = name.replace('Á', 'A').replace('Í', 'I').replace('Ú', 'U')
    # name = name.replace('ṇ', 'n').replace('ṇa', 'na')
    # name = name.replace('ṛ', 'r').replace('ṃ', 'm').replace('ś', 's')
    
    # Remove any remaining special characters but keep the name structure
    name = re.sub(r'[^\w\s-]', '', name)
    
    # Clean up whitespace
    name = ' '.join(name.split())
    
    return name

def is_valid_character_name(name):

    if name.lower() in STOP_WORDS:
        return False

    # Remove common false positives
    false_positives = {
        'shall', 'sweet', 'brave', 'fair', 'true', 'good', 'great',
        'wise', 'holy', 'best', 'first', 'last', 'next', 'each',
        'both', 'all', 'some', 'many', 'few', 'more', 'most',
        'king', 'queen', 'prince', 'lord', 'lady', 'god', 'goddess',
        'performed', 'moon',
    }
    
    # Check if it's a false positive
    if name.lower() in false_positives:
        return False
    
    # Must be at least 3 characters
    if len(name) < 3:
        return False
    
    # Should start with capital letter
    if not name[0].isupper():
        return False
    
    # Should not be all uppercase (likely acronym or title)
    if name.isupper() and len(name) > 1:
        return False
    
    return True

def extract_characters_ner(text, min_frequency=3):
    doc = nlp(text)
    
    # Extract PERSON entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    # Normalize names
    normalized_persons = []
    for person in persons:
        normalized = normalize_name(person)
        if normalized and is_valid_character_name(normalized):
            normalized_persons.append(normalized)
    
    # Count frequencies
    person_counts = Counter(normalized_persons)
    
    # Filter by minimum frequency
    characters = [name for name, count in person_counts.items() if count >= min_frequency]
    
    # Sort by frequency (most common first)
    characters = sorted(characters, key=lambda x: person_counts[x], reverse=True)
    
    return characters, person_counts

# Extract characters from all cantos combined
print("Extracting characters using NER...")
all_text = ' '.join(cantos)
characters, char_counts = extract_characters_ner(all_text, min_frequency=3)

print(f"\nFound {len(characters)} valid characters appearing at least 3 times:")
for char in characters:
    print(f"  {char}: {char_counts[char]} mentions")
print()

# ---- Build co-occurrence pairs ----
pair_counts = Counter()

for i, canto_text in enumerate(cantos, 1):
    # Find characters appearing in this canto (search for normalized versions)
    present = []
    print(canto_text)
    for char in characters:
        # Create pattern that matches the character with various diacritics
        pattern = char
        # Make pattern flexible for diacritics
        pattern = pattern.replace('a', '[aá]').replace('i', '[ií]').replace('u', '[uú]')
        pattern = pattern.replace('n', '[nṇ]')
        
        if re.search(rf"\b{pattern}\b", canto_text, re.IGNORECASE):
            present.append(char)
    
    print(f"Canto {i}: {len(present)} characters found - {present[:5]}{'...' if len(present) > 5 else ''}")
    
    for a, b in itertools.combinations(sorted(present), 2):
        pair_counts[(a, b)] += 1

print(f"\nTotal unique character pairs: {len(pair_counts)}\n")

# ---- Create NetworkX graph ----
G = nx.Graph()
for (a, b), w in pair_counts.items():
    G.add_edge(a, b, weight=w)

# Node size by total appearances
node_weight = defaultdict(int)
for (a, b), w in pair_counts.items():
    node_weight[a] += w
    node_weight[b] += w

if len(G.nodes()) > 0:
    sizes = [node_weight[n]*200 for n in G.nodes()]
    
    # ---- Visualize ----
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="lightblue", edgecolors="black")
    nx.draw_networkx_edges(G, pos, width=[G[u][v]['weight']*0.5 for u,v in G.edges()], alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    plt.title("Character Co-occurrence Network (First 10 Cantos of Ramayana) - NER Based", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    
    # ---- Print top co-occurring pairs ----
    print("Top Co-occurring Pairs:")
    for (a, b), w in pair_counts.most_common(15):
        print(f"{a} - {b}: {w}")
    
    # ---- Print node statistics ----
    print("\nCharacter Appearance Statistics:")
    for char in sorted(node_weight.keys(), key=lambda x: node_weight[x], reverse=True):
        print(f"{char}: {node_weight[char]} co-occurrences")
else:
    print("No character relationships found. Try lowering min_frequency.")