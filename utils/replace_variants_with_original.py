import json
import re

# Load the input files
try:
    with open('/Users/chetanvellanki/Desktop/ramayana-network-analysis/data/ramayana_resolved.json', 'r') as f:
        ramayana_data = json.load(f)
    with open('/Users/chetanvellanki/Desktop/ramayana-network-analysis/data/name_alias_mapping.json', 'r') as f:
        alias_mapping_data = json.load(f)
except FileNotFoundError as e:
    print(f"Error: Required file not found: {e.filename}")
    exit()
except json.JSONDecodeError:
    print("Error: Failed to decode JSON from one of the input files.")
    exit()

# 1. Create a reverse mapping from variant (alias) to canonical name
# { "variant_name": "Canonical Name" }
variant_to_canonical = {}
for canonical_name, variants in alias_mapping_data.items():
    for variant in variants:
        # Only map the variant if it's different from the canonical name
        # We process the canonical name separately later to ensure consistency.
        if variant != canonical_name:
             variant_to_canonical[variant] = canonical_name

# 2. Function to process the text and replace variants
def resolve_text(text):
    # Ensure replacements are done on full words using regex word boundaries (\b)
    # Sort by length descending to ensure longer variants (like compounds) are
    # replaced before their substrings are mistakenly replaced first.
    sorted_variants = sorted(variant_to_canonical.keys(), key=len, reverse=True)

    for variant in sorted_variants:
        canonical_name = variant_to_canonical[variant]
        
        # Regex to find the whole word variant
        # re.escape is used in case the variant contains regex special characters
        pattern = re.compile(r'\b' + re.escape(variant) + r'\b')
        
        # Replace all occurrences of the variant with the canonical name
        text = pattern.sub(canonical_name, text)
        
    # Also ensure the canonical name itself is normalized (important for self-aliases)
    # The canonical names might appear in various casing in the original text,
    # so we should use the version from the key of the alias_mapping_data for the final canonical form.
    for canonical_name in alias_mapping_data.keys():
        # Handle cases where the canonical name itself might appear in a variant's spot
        # or inconsistent casing is used for the canonical name in the input text.
        
        # 1. Replace all canonical name variations found in the list with the target casing
        for variant in alias_mapping_data[canonical_name]:
            if variant == canonical_name:
                continue # Skip the perfect match for the first loop to prevent infinite recursion/issues
            
            # Use regex for replacement to ensure word boundaries
            pattern = re.compile(r'\b' + re.escape(variant) + r'\b')
            text = pattern.sub(canonical_name, text)
        
        # 2. Finally, normalize the casing of the canonical name itself (e.g., "rama" -> "Rama")
        # This part assumes that the key in the mapping file is the desired final casing.
        if canonical_name in text:
            # Create a list of all casing variations of the canonical name itself (e.g., Rama, RAMA, rama)
            # and replace them with the exact canonical_name casing.
            name_variations = set([canonical_name.lower(), canonical_name.upper(), canonical_name])
            
            for variation in name_variations:
                if variation != canonical_name:
                    pattern = re.compile(r'\b' + re.escape(variation) + r'\b')
                    text = pattern.sub(canonical_name, text)

    return text


# 3. Process the entire text structure
resolved_ramayana_data = ramayana_data.copy()

for chapter in resolved_ramayana_data['chapters']:
    # The assumption is that 'cantos' contains a list of strings (the text)
    # If 'cantos' is a list of dictionaries, this needs adjustment. 
    # Based on the provided structure, it's a list of strings.
    if 'cantos' in chapter:
        for i, canto_text in enumerate(chapter['cantos']):
            chapter['cantos'][i] = resolve_text(canto_text)

# 4. Save the resolved data to a new JSON file
output_file_name = 'ramayana_resolved_final.json'
with open(output_file_name, 'w') as f:
    json.dump(resolved_ramayana_data, f, indent=4)

print("✅ Character name resolution complete.")
print(f"The resolved text has been saved to: {output_file_name}")

# Optional: Print a small snippet for verification
if resolved_ramayana_data['chapters']:
    first_canto = resolved_ramayana_data['chapters'][0]['cantos'][0]
    print("\n--- Snippet of Resolved Text (First Canto) ---")
    print(first_canto[:1000] + "...")