import re
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

with open("ramayan_trunc.txt", "r", encoding="utf-8") as f:
    text = f.read()

das_count = 0
def extract_books_and_cantos(text, num_books=None, cantos_per_book=None):
    # First, split by books
    book_pattern = re.compile(
        r'BOOK\s+([IVXLC]+)\.?\s*(.*?)(?=BOOK\s+[IVXLC]+\.?|\Z)',
        re.DOTALL | re.IGNORECASE
    )
    book_matches = book_pattern.findall(text)
    
    print(f"Found {len(book_matches)} books in the text")
    
    # Limit books if specified
    if num_books:
        book_matches = book_matches[:num_books]
        print(f"Analyzing first {num_books} books")
    
    all_cantos = []
    
    # Roman numeral to integer conversion
    def roman_to_int(roman):
        roman_values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        total = 0
        prev_value = 0
        for char in reversed(roman.upper()):
            value = roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        return total
    
    # Extract cantos from each book
    for book_idx, (book_roman, book_text) in enumerate(book_matches, 1):
        book_num = roman_to_int(book_roman)
        
        # Extract cantos from this book
        canto_pattern = re.compile(
            r'Canto\s+([IVXLC]+)\.\s*(.*?)(?=Canto\s+[IVXLC]+\.|\Z)',
            re.DOTALL | re.IGNORECASE
        )
        canto_matches = canto_pattern.findall(book_text)
        
        print(f"  Book {book_roman} ({book_num}): Found {len(canto_matches)} cantos")
        
        # Limit cantos per book if specified
        if cantos_per_book:
            canto_matches = canto_matches[:cantos_per_book]
        
        # Add cantos with book information
        for canto_roman, canto_text in canto_matches:
            canto_num = roman_to_int(canto_roman)
            # Replace newlines with spaces in canto text
            canto_text_clean = canto_text.replace('\n', ' ')
            global das_count
            das_count += canto_text_clean.count("Daśaratha")
            all_cantos.append((book_num, book_roman, canto_num, canto_roman, canto_text_clean))
    
    return all_cantos

# Configuration
NUM_BOOKS_TO_ANALYZE = 1  # Analyze first 2 books (change to None for all)
NUM_CANTOS_PER_BOOK = 20  # Take 10 cantos from each book (change to None for all)

# Extract books and cantos
structured_cantos = extract_books_and_cantos(text, NUM_BOOKS_TO_ANALYZE, NUM_CANTOS_PER_BOOK)
print(das_count)

print(f"\nTotal cantos to analyze: {len(structured_cantos)}")
# print("\nFirst few cantos:")
# for book_num, book_roman, canto_num, canto_roman, canto_text in structured_cantos[:3]:
#     preview = canto_text[:100].replace('\n', ' ')
#     print(f"  Book {book_roman}.{canto_roman}: {preview}...")

try:
    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 3000000
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
        'king', 'queen', 'prince', 'lord', 'lady', 'god', 'goddess', 'performed', 'moon'
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
print("\nExtracting characters using NER with stopword removal...")
all_canto_texts = [canto_text for _, _, _, _, canto_text in structured_cantos]
combined_text = ' '.join(all_canto_texts)
characters, char_counts = extract_characters_ner(combined_text, min_frequency=3)

print(f"\nFound {len(characters)} valid characters appearing at least 3 times:")
for char in characters:
    print(f"  {char}: {char_counts[char]} mentions")
print()

# ---- Build co-occurrence pairs ----
pair_counts = Counter()

for book_num, book_roman, canto_num, canto_roman, canto_text in structured_cantos:
    present = []
    for char in characters:
        pattern = char
        pattern = pattern.replace('a', '[aá]').replace('i', '[ií]').replace('u', '[uú]')
        pattern = pattern.replace('n', '[nṇ]').replace('r', '[rṛ]').replace('s', '[sś]')
        
        if re.search(rf"\b{pattern}\b", canto_text, re.IGNORECASE):
            present.append(char)
    
    print(f"Book {book_roman}, Canto {canto_roman}: {len(present)} characters")
    
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
    plt.title("Character Co-occurrence Network (First 20 Cantos of Ramayana) - NER Based", fontsize=14)
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