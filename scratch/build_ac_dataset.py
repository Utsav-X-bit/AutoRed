import bz2
import json
import re
import os
from collections import Counter

INPUT_FILE = "experiment/raw_dump_defenses.jsonl.bz2"
OUTPUT_FILE = "data/access_code_classifier_dataset.jsonl"

def classify_access_code(code: str) -> str:
    if not code:
        return "TOKEN"  # Fallback for empty
    
    code = code.strip()
    
    # 1. MULTILINE
    if "\n" in code or len(code) > 150:
        return "MULTILINE"
        
    words = code.split()
    word_count = len(words)
    
    # 2. SENTENCE
    # Has punctuation typical of a sentence and is relatively long
    has_punctuation = bool(re.search(r'[.!?]', code))
    if word_count > 7 and has_punctuation:
        return "SENTENCE"
        
    # 3. PHRASE
    if 2 <= word_count <= 7:
        return "PHRASE"
        
    # 4. TOKEN
    # word_count == 1, or very short without spaces
    return "TOKEN"

def build_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return
        
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    stats = Counter()
    processed = 0
    skipped = 0
    
    print(f"Processing {INPUT_FILE}...")
    
    with bz2.open(INPUT_FILE, "rt", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        
        for line in fin:
            try:
                data = json.loads(line)
                opening = (data.get("opening_defense") or "").strip()
                closing = (data.get("closing_defense") or "").strip()
                access_code = (data.get("access_code") or "").strip()
                
                if not opening and not closing:
                    skipped += 1
                    continue
                    
                ac_type = classify_access_code(access_code)
                stats[ac_type] += 1
                
                out_data = {
                    "opening_defense": opening,
                    "closing_defense": closing,
                    "access_code_type": ac_type
                }
                
                fout.write(json.dumps(out_data) + "\n")
                processed += 1
                
                if processed % 20000 == 0:
                    print(f"Processed {processed} records...")
                    
            except json.JSONDecodeError:
                skipped += 1
                continue
                
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total processed: {processed}")
    print(f"Total skipped: {skipped}")
    print(f"Output saved to: {OUTPUT_FILE}")
    print("\nClass Distribution:")
    
    for ac_type, count in stats.most_common():
        pct = (count / processed) * 100
        print(f"  {ac_type:10}: {count:7} ({pct:.1f}%)")

if __name__ == "__main__":
    build_dataset()
