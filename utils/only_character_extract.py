import json

# Define the input and output file names
FILE_MAPPING = '/Users/chetanvellanki/Desktop/ramayana-network-analysis/data/name_alias_mapping.json'
OUTPUT_FILE = 'canonical_characters.json'

# 1. Load the alias mapping file
try:
    with open(FILE_MAPPING, 'r') as f:
        alias_mapping_data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{FILE_MAPPING}' was not found.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Failed to decode JSON from '{FILE_MAPPING}'.")
    exit()

# 2. Extract canonical character names (the keys of the dictionary)
canonical_names = list(alias_mapping_data.keys())

# 3. Save the list to a new JSON file
with open(OUTPUT_FILE, 'w') as f:
    json.dump(canonical_names, f, indent=4)

print(f"Extraction complete. Canonical names saved to: {OUTPUT_FILE}")
print(f"Total characters extracted: {len(canonical_names)}")