import json
import re
from collections import defaultdict


def clean_english(text):
    clean_pattern = re.compile(r"[^A-Za-z0-9\s.,;:!?()'\"-]")
    return re.sub(clean_pattern, "", text)

# Load input JSON
with open("../data/Valmiki_Ramayan_Shlokas.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Structure: { kanda: { sarga: [explanations...] } }
grouped = defaultdict(lambda: defaultdict(list))

for shloka in data:
    kanda = shloka["kanda"]
    sarga = shloka["sarga"]
    explanation = shloka.get("explanation", "")
    if explanation:
        grouped[kanda][sarga].append(explanation.strip())

result = {"chapters": []}

for kanda, sargas in grouped.items():
    cantos = []
    for sarga_num, expl_list in sorted(sargas.items()):
        english_text = " ".join(expl_list)
        cantos.append(clean_english(english_text))
    result["chapters"].append({
        "kanda": kanda,
        "cantos": cantos
    })

with open("../data/ramayana_compiled.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=4)

print("✅ Grouped JSON created: ramayana_grouped.json")
