import json
from collections import OrderedDict
import pathlib

IN_PATH = "dataset/principal_names.txt"
OUT_JSON = "dataset/principal_names_list.json"

names = []
with open(IN_PATH, "r", encoding="utf-8") as f:
    for raw in f:
        line = raw.strip()
        if not line:
            continue  # skip empty lines

        # split on first comma (or whitespace)
        first = line.split(",", 1)[0].strip() if "," in line else line.split(None, 1)[0].strip()
        first = first.strip(" .;:()[]\"'")

        # filter: must start alphabetic and not contain '_'
        if first and first[0].isalpha() and "_" not in first:
            names.append(first)

# deduplicate while preserving order
names = list(OrderedDict.fromkeys(names))

# save to JSON
pathlib.Path(OUT_JSON).write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")

print(f"✅ Extracted {len(names)} names. Sample:")
print(names[:30])
