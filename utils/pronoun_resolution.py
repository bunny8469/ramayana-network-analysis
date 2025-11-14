import json
import time
from google import genai
from tqdm import tqdm

# -------------------------- CONFIG --------------------------
API_KEY = "AIzaSyCKK2UcH2gM8gboM4WLIBxmVGcZAt_BqNU"
MODEL_NAME = "gemini-2.5-flash-lite"

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are given a canto from the Ramayana.

Your PRIMARY task:
1. Replace every occurrence of gendered pronouns 
   (he, him, his, she, her, hers) 
   with the correct character name based strictly on the context 
   of the canto and the Ramayana story.

This replacement is the highest priority. Do not skip any occurrences.

After that, your SECONDARY task:
2. If a pronoun appears as part of a relational phrase 
   such as "his father", "her husband", "his brother", etc.,
   replace the whole phrase with the correct specific name 
   ONLY if the relationship is canonically known and unambiguous.

Examples:
- "his father" → "King Dasaratha" (if referring to Rama)
- "her husband" → "Rama" (if referring to Sita)
- "his brother" → "Lakshmana" (if referring to Rama)

If the relationship is not identifiable with high confidence, 
leave the phrase exactly as it appears.

RULES:
- Use only canonical Ramayana character names.
- Do not invent or add story content.
- Maintain original sentence structure.
- Output ONLY the transformed canto text.
"""

# -------------------------------------------------------------

def process_canto(canto_text):
    """Send a single canto to Gemini."""
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=SYSTEM_PROMPT + "\n\nCANTO:\n" + canto_text
        )
        return response.text
    except Exception as e:
        print("Error:", e)
        return canto_text  # fallback


def save_progress(output_path, new_data):
    """Write progress safely to output file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(f"💾 Auto-saved progress → {output_path}")


def process_dataset(input_path, output_path):
    """Process dataset with chapter-wise progress bars & auto-save every 20 cantos."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = {"chapters": []}
    canto_counter = 0  # global counter for autosave

    for chapter_index, chapter in enumerate(data["chapters"]):
        print(f"\n📘 Processing Chapter {chapter_index+1}: {chapter['kanda']}")
        new_chapter = {"kanda": chapter["kanda"], "cantos": []}

        # Create progress bar for this chapter
        for canto in tqdm(chapter["cantos"], desc=f"Chapter {chapter_index+1} Progress", unit="canto"):
            canto_counter += 1

            cleaned = process_canto(canto)
            new_chapter["cantos"].append(cleaned)

            # Auto-save every 20 cantos
            if canto_counter % 20 == 0:
                new_data["chapters"].append(new_chapter)
                save_progress(output_path, new_data)
                new_data["chapters"].pop()  # Remove last to avoid duplicates

            time.sleep(1.5)

        # Append final chapter results
        new_data["chapters"].append(new_chapter)

    # Final save
    save_progress(output_path, new_data)
    print("\n🎉 All chapters processed successfully!")


# Run the script
process_dataset("../data/ramayana_compiled.json", "../data/ramayana_resolved.json")
