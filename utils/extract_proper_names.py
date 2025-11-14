import json
import time
from google import genai
from tqdm import tqdm
import re

# -------------------------- CONFIG --------------------------
API_KEY = "AIzaSyBK3FiOQpgN8I58kVRxceryrx4pekQwQKQ"
MODEL_NAME = "gemini-2.5-flash-lite"

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are given a canto from the Ramayana.

Your task:
Extract all proper nouns appearing in the canto text.

Proper nouns include:
- Characters (Rama, Sita, Lakshmana, Ravana…)
- Places (Ayodhya, Lanka…)
- Deities or mythological entities (Indra, Agni…)
- Any meaningful capitalized term that refers to a person, place, or entity.

RULES:
- Do NOT normalize or change spellings. Keep every name exactly as it appears.
- Do NOT add names not present in the canto.
- Do NOT explain or comment.
- Output ONLY a JSON array of strings. Example:
  ["Rama", "Lakshmana", "Ayodhya"]
"""

# -------------------------------------------------------------

def extract_names(canto_text):
    try:
        prompt = SYSTEM_PROMPT + "\n\nCANTO:\n" + canto_text

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt]
        )

        # Get text reliably
        try:
            raw = response.text
        except:
            try:
                raw = response.candidates[0].content.parts[0].text
            except:
                print("⚠️ No text output:", response)
                return []

        raw = raw.strip()

        # --- FIXED CLEANING LOGIC ---
        # Remove leading ```json or ``` but keep content
        raw = re.sub(r"^```(?:json)?", "", raw.strip())
        raw = re.sub(r"```$", "", raw.strip())
        raw = raw.strip()

        # --- 1. Try direct JSON parse ---
        try:
            return json.loads(raw)
        except:
            pass

        # --- 2. Extract JSON array from messy text ---
        match = re.search(r"\[[\s\S]*\]", raw)
        if match:
            try:
                return json.loads(match.group(0))
            except:
                pass

        print("⚠️ Could not parse model output:\n", raw)
        return []

    except Exception as e:
        print("Error:", e)
        return []


def process_dataset(input_path, output_path):
    """Iterate cantos → accumulate proper nouns → save JSON."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    proper_noun_set = set()

    for chapter_index, chapter in enumerate(data["chapters"]):
        print(f"\n📘 Processing Chapter {chapter_index+1}: {chapter['kanda']}")

        for canto in tqdm(
            chapter["cantos"], desc=f"Chapter {chapter_index+1} Progress", unit="canto"
        ):
            names = extract_names(canto)

            # Add names to the global set
            for name in names:
                proper_noun_set.add(name)

            time.sleep(3)

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted(list(proper_noun_set)), f, ensure_ascii=False, indent=4)

    print(f"\n🎉 Extracted {len(proper_noun_set)} unique proper nouns → {output_path}")


# ---------------- RUN SCRIPT ----------------
process_dataset(
    input_path="../data/ramayana_compiled.json", output_path="../data/proper_nouns.json"
)
