import json

# Load the two JSON files
with open("../data/name_alias_mapping.json", "r") as f:
    mappings = json.load(f)

with open("../data/proper_nouns_cleaned.json", "r") as f:
    names_list = json.load(f)

# Collect all canonical names + all their aliases
all_mapping_names = set(mappings.keys())
all_aliases = set()

for canonical, aliases in mappings.items():
    all_aliases.add(canonical)
    for a in aliases:
        all_aliases.add(a)

# Now check
names_set = set(names_list)

# Names that are in names.json but not in the mapping (either as main name or alias)
missing_in_mappings = sorted(names_set - all_aliases)

# Names that exist in the mappings (main or alias) but not present in names.json
extra_in_mappings = sorted(all_aliases - names_set)

print("=== Missing in mappings (names not found) ===")
for name in missing_in_mappings:
    print(name)

print("\n=== Extra in mappings (not present in names list) ===")
for name in extra_in_mappings:
    print(name)
