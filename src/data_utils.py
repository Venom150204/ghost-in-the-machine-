"""
Data utilities for Gutenberg text cleaning, chunking, and dataset assembly.
"""

import re
import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from tqdm import tqdm


def clean_gutenberg_text(filepath: str) -> str:
    """
    Strips Project Gutenberg boilerplate and cleans the raw text while
    preserving all original punctuation (punctuation is a feature, not noise).

    Steps:
    1. Remove everything before '*** START OF' and after '*** END OF'
    2. Remove chapter headings and standalone Roman numerals
    3. Remove lines that are entirely uppercase (section markers)
    4. Collapse multiple whitespace into single spaces within paragraphs
    5. Preserve paragraph boundaries (double newlines)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Strip Gutenberg header and footer
    start_match = re.search(r"\*\*\* START OF .+? \*\*\*", raw)
    end_match = re.search(r"\*\*\* END OF .+? \*\*\*", raw)

    if start_match:
        raw = raw[start_match.end():]
    # Re-search for END marker on the already-trimmed text
    end_match2 = re.search(r"\*\*\* END OF .+? \*\*\*", raw)
    if end_match2:
        raw = raw[:end_match2.start()]

    lines = raw.split("\n")
    cleaned_lines = []

    # Patterns to remove
    # Chapter headings: "Chapter I", "CHAPTER XII.", "Letter 1", etc.
    chapter_pat = re.compile(
        r"^\s*(chapter|letter|volume|vol\.?|book|part)\s+"
        r"[ivxlcdm\d]+\.?\s*$",
        re.IGNORECASE,
    )
    # Standalone Roman numerals: "  IV  " or "  XII.  "
    roman_pat = re.compile(
        r"^\s*[IVXLCDM]+\.?\s*$"
    )
    # Lines like "CONTENTS", "INTRODUCTION.", table-of-contents entries
    contents_pat = re.compile(r"^\s*CONTENTS\s*$", re.IGNORECASE)
    # TOC entries: "I. In Chancery", "XIV. Deportment", etc.
    toc_entry_pat = re.compile(r"^\s*[IVXLCDM]+\.\s+\S")
    # Standalone section words
    section_word_pat = re.compile(
        r"^\s*(Preface|Dedication|Appendix|Epilogue|Prologue|Foreword"
        r"|Introduction|Conclusion)\s*\.?\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        stripped = line.strip()

        # Skip blank lines but preserve them as paragraph boundaries
        if not stripped:
            cleaned_lines.append("")
            continue

        # Skip all-caps lines (section markers like "VOL. I.", "LONDON:")
        if stripped.isupper() and len(stripped) > 1:
            continue

        # Skip chapter/letter/volume headings
        if chapter_pat.match(stripped):
            continue

        # Skip standalone Roman numerals
        if roman_pat.match(stripped):
            continue

        # Skip contents line
        if contents_pat.match(stripped):
            continue

        # Skip TOC entries (Roman numeral + title)
        if toc_entry_pat.match(stripped):
            continue

        # Skip standalone section words
        if section_word_pat.match(stripped):
            continue

        cleaned_lines.append(stripped)

    # Rejoin and collapse multiple blank lines into double newlines
    text = "\n".join(cleaned_lines)
    # Collapse 3+ newlines into 2 (paragraph boundary)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces within lines
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def chunk_into_paragraphs(text: str, min_words: int = 100, max_words: int = 200) -> list[str]:
    """
    Splits cleaned text on double newlines (Gutenberg paragraph boundaries)
    and keeps only chunks within the word count range.

    Short paragraphs (< min_words) are merged with the next paragraph to
    avoid discarding too much text. Chunks exceeding max_words are split
    at sentence boundaries where possible.
    """
    raw_paragraphs = re.split(r"\n\n+", text)

    # Merge very short consecutive paragraphs (dialogue lines, etc.)
    merged = []
    buffer = ""
    for para in raw_paragraphs:
        para = para.replace("\n", " ").strip()
        if not para:
            continue
        if buffer:
            buffer = buffer + " " + para
        else:
            buffer = para

        word_count = len(buffer.split())
        if word_count >= min_words:
            merged.append(buffer)
            buffer = ""

    # Don't lose the trailing buffer if it meets minimum
    if buffer and len(buffer.split()) >= min_words:
        merged.append(buffer)

    # Filter to [min_words, max_words] range
    chunks = []
    for para in merged:
        words = para.split()
        wc = len(words)
        if min_words <= wc <= max_words:
            chunks.append(para)
        elif wc > max_words:
            # Split at roughly max_words boundary, respecting sentence ends
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = []
            current_wc = 0
            for sent in sentences:
                sent_wc = len(sent.split())
                if current_wc + sent_wc > max_words and current_wc >= min_words:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sent]
                    current_wc = sent_wc
                else:
                    current_chunk.append(sent)
                    current_wc += sent_wc
            if current_chunk and min_words <= current_wc <= max_words:
                chunks.append(" ".join(current_chunk))

    return chunks


def generate_gemini_paragraphs(
    topics: list[str],
    prompt_template: str,
    total_count: int,
    output_path: str,
    model_name: str = "gemini-2.5-flash",
    rate_limit_sleep: float = 0.1,
    temperature: float = 0.8,
    max_output_tokens: int = 300,
    top_p: float = 0.9,
    max_workers: int = 10,
) -> list[dict]:
    """
    Generates paragraphs via the Gemini API, distributing evenly across topics.
    Uses concurrent workers for speed. Saves results incrementally to disk (crash-safe).

    Args:
        temperature: Sampling temperature (higher = more diverse).
        max_output_tokens: Max tokens per response.
        top_p: Nucleus sampling threshold.
        max_workers: Number of concurrent API workers.

    Returns list of dicts: {text, topic, word_count}
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    print(f"  Model: {model_name}")
    print(f"  API key: {'set (' + api_key[:8] + '...)' if api_key else 'MISSING!'}")
    print(f"  Generation config: temp={temperature}, top_p={top_p}, max_tokens={max_output_tokens}")
    print(f"  Concurrency: {max_workers} workers")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    gen_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
    )

    # Quick test call to verify the model works before starting the loop
    print("  Testing API connection...", flush=True)
    try:
        test_resp = model.generate_content("Say hello in one word.")
        print(f"  API test OK: '{test_resp.text.strip()[:50]}'")
    except Exception as e:
        print(f"  API test FAILED: {type(e).__name__}: {e}")
        raise

    # If output file already exists, resume from where we left off
    results = []
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming: {len(results)} paragraphs already generated.")

    per_topic = total_count // len(topics)
    remainder = total_count % len(topics)

    # Count how many we already have per topic
    existing_counts = {}
    for r in results:
        existing_counts[r["topic"]] = existing_counts.get(r["topic"], 0) + 1

    # Build the full work list: (topic, prompt) pairs for all remaining paragraphs
    work_items = []
    for i, topic in enumerate(topics):
        target = per_topic + (1 if i < remainder else 0)
        already_done = existing_counts.get(topic, 0)
        needed = target - already_done
        for _ in range(needed):
            work_items.append(topic)

    if not work_items:
        print("All paragraphs already generated.")
        return results

    print(f"Generating {len(work_items)} paragraphs across {len(topics)} topics...")

    # Thread-safe lock for results list and file saving
    lock = threading.Lock()
    save_counter = [0]  # mutable counter for incremental saves

    def _generate_one(topic):
        """Generate a single paragraph with retry logic."""
        prompt = prompt_template.format(topic_name=topic)
        for attempt in range(5):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=gen_config,
                    request_options={"timeout": 60},
                )
                text = response.text.strip()
                time.sleep(rate_limit_sleep)
                return {"text": text, "topic": topic, "word_count": len(text.split())}
            except Exception as e:
                wait = rate_limit_sleep * (2 ** attempt) + 1
                print(f"  API error (attempt {attempt+1}/5): {type(e).__name__}. Retrying in {wait:.0f}s...")
                time.sleep(wait)
        print(f"  FAILED after 5 retries on topic='{topic}', skipping.")
        return None

    # Use ThreadPoolExecutor for concurrent generation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_generate_one, topic): topic for topic in work_items}
        pbar = tqdm(total=len(work_items), desc="Generating")

        for future in as_completed(futures):
            result = future.result()
            if result:
                with lock:
                    results.append(result)
                    save_counter[0] += 1
                    # Save incrementally every 10 paragraphs
                    if save_counter[0] % 10 == 0:
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)
            pbar.update(1)
        pbar.close()

    # Final save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Total paragraphs generated: {len(results)}")
    return results


def assemble_dataset(
    human_chunks: dict[str, list[str]],
    class2_path: str,
    class3_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Merges human paragraphs and AI-generated paragraphs into a single CSV.

    Columns: text, class_label, class_name, source_author_or_model, topic, word_count
    - class_label 0 = Human, 1 = Generic AI, 2 = Style-Mimicking AI
    """
    rows = []

    # Human paragraphs (Class 0)
    for author, chunks in human_chunks.items():
        for chunk in chunks:
            rows.append({
                "text": chunk,
                "class_label": 0,
                "class_name": "Human",
                "source_author_or_model": author,
                "topic": "original_text",
                "word_count": len(chunk.split()),
            })

    # Class 2: Generic AI
    with open(class2_path, "r", encoding="utf-8") as f:
        class2_data = json.load(f)
    for item in class2_data:
        rows.append({
            "text": item["text"],
            "class_label": 1,
            "class_name": "Generic AI",
            "source_author_or_model": "gemini",
            "topic": item["topic"],
            "word_count": item["word_count"],
        })

    # Class 3: Style-Mimicking AI
    with open(class3_path, "r", encoding="utf-8") as f:
        class3_data = json.load(f)
    for item in class3_data:
        rows.append({
            "text": item["text"],
            "class_label": 2,
            "class_name": "Style-Mimicking AI",
            "source_author_or_model": "gemini_style_mimic",
            "topic": item["topic"],
            "word_count": item["word_count"],
        })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved: {len(df)} rows -> {output_path}")
    print(f"Class distribution:\n{df['class_name'].value_counts()}")
    return df