import os
import re
import time
import json
import traceback
from pathlib import Path
from string import Template

import db.db_tools as db
from json_helpers import json_cleaner
from mistral.mistral_interface import load_mistral, generate_text
from context_calc.context_slider import strip_consumed  # fuzzy strip helper

INPUT_FILE = Path(os.environ.get("INPUT_FILE", "nameofthewind.txt"))
DB_FILE = Path(os.environ.get("DB_FILE", "paragraphs_mistral.db"))

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 1600))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 2))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))

SAVE_FAILED_DIR = Path(os.environ.get("SAVE_FAILED_DIR", "failed_chunks"))
SAVE_FAILED_DIR.mkdir(exist_ok=True)

HISTORY_LOC = Path(os.environ.get("HISTORY_LOC", "history_loc"))
HISTORY_LOC.mkdir(exist_ok=True)

VERBOSE = True

PROMPT_TEMPLATE = Template("""Return ONLY one valid JSON object in this form:

{ "type": "header"|"paragraph", "content": "..." }

Rules:
- Output exactly ONE logical block (the first complete block in the text).
- Titles/headings = "header".
- Paragraphs, lists, and publisher/legal info = "paragraph".
- Group related lines together into one block; ignore raw line breaks.
- Collapse extra whitespace into single spaces.

Text:

""" + '"""' + "${chunk}" + '"""')


def info(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)

def save_raw_output(raw: str, chunk_num: int, suffix: str = "raw", prompt: str = None) -> Path:
    ts = int(time.time())
    raw_fname = HISTORY_LOC / f"chunk_{chunk_num}_{suffix}_{ts}.txt"
    with open(raw_fname, "w", encoding="utf-8") as fh:
        fh.write(raw)
    if prompt is not None:
        prompt_fname = HISTORY_LOC / f"chunk_{chunk_num}_{suffix}_prompt_{ts}.txt"
        with open(prompt_fname, "w", encoding="utf-8") as fh:
            fh.write(prompt)
    return raw_fname

def safe_save_failed(raw: str, chunk_num: int) -> Path:
    fname = SAVE_FAILED_DIR / f"failed_chunk_{chunk_num}_{int(time.time())}.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(raw)
    return fname

def normalize_for_search(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def parse_single_json(s: str):
    """Extract and parse the first JSON object from a string."""
    # fast path
    try:
        return json.loads(s)
    except Exception:
        pass
    # cleaned quotes
    try:
        cleaned = json_cleaner.normalize_quotes(s)
        return json.loads(cleaned)
    except Exception:
        pass
    # minimal brace capture
    m = re.search(r'\{.*?\}', s, flags=re.DOTALL)
    if m:
        blob = m.group(0)
        for candidate in (blob, json_cleaner.normalize_quotes(blob)):
            try:
                return json.loads(candidate)
            except Exception:
                continue
    return None

# ---- single-table chapter/block counters ----
current_chapter_num = 0                 # 0 = prologue until first header
chapter_block_counter = {}              # chapter_num -> next block_num

def next_block_num(ch: int) -> int:
    n = chapter_block_counter.get(ch, 1)
    chapter_block_counter[ch] = n + 1
    return n

def main():
    # DB init
    db.set_db_file(str(DB_FILE))
    db.init()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    info("[info] loading model/tokenizer")
    tokenizer, model, eos_id = load_mistral()
    info("[info] model loaded. eos_id:", eos_id)

    full_text = INPUT_FILE.read_text(encoding="utf-8")
    # join hyphen-split words and normalize newlines
    full_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", full_text)
    full_text = full_text.replace("\r\n", "\n")

    info(f"[info] input length {len(full_text)} chars")

    chunk_num = 1

    # Consume from the front until empty
    while len(full_text) > 0:
        window = full_text[:WINDOW_SIZE]
        if not window.strip():
            break

        prompt = PROMPT_TEMPLATE.safe_substitute(chunk=window)

        parsed = None
        raw_out = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                info(f"[chunk {chunk_num}] generating (attempt {attempt}) prompt len={len(prompt)}")
                raw_out = generate_text(tokenizer, model, eos_id, prompt, temperature=TEMPERATURE)
                info(f"[chunk {chunk_num}] raw preview:\n{raw_out[:800].replace(chr(10), ' ')}\n")

                # Save raw + prompt
                save_raw_output(raw_out, chunk_num, suffix=f"attempt{attempt}", prompt=prompt)

                parsed = parse_single_json(raw_out)
                if isinstance(parsed, dict) and "content" in parsed:
                    break

                time.sleep(0.1)  # tiny backoff between retries

            except Exception as e:
                info(f"[error] generation exception on chunk {chunk_num}: {e}")
                traceback.print_exc()
                save_raw_output(f"EXC:{e}\n{traceback.format_exc()}", chunk_num, suffix="exception", prompt=prompt)

        if not (isinstance(parsed, dict) and "content" in parsed):
            info(f"[x] skipping chunk {chunk_num} — no valid JSON object")
            # Safety advance to avoid infinite loop
            full_text = full_text[WINDOW_SIZE // 2 :]
            chunk_num += 1
            continue

        btype = (parsed.get("type") or "").lower()
        content = (parsed.get("content") or "").strip()

        if not content:
            info(f"[x] empty content; skipping chunk {chunk_num}")
            full_text = full_text[WINDOW_SIZE // 2 :]
            chunk_num += 1
            continue

        # ---------- DB persistence (single-table) ----------
        global current_chapter_num

        if btype == "header":
            current_chapter_num += 1
            db.insert_block(current_chapter_num, next_block_num(current_chapter_num), "header", content)
            info(f"[+] Chapter {current_chapter_num}: {content[:120]}")

        else:
            # ensure we have a chapter (use 0 until first header)
            if current_chapter_num not in chapter_block_counter:
                chapter_block_counter[current_chapter_num] = 1
                # optional: create a synthetic prologue header once
                # db.insert_block(0, next_block_num(0), "header", "Prologue/Intro")

            # optional dedupe
            if btype == "paragraph" and db.has_block(content):
                info("[skip] paragraph already exists in DB (dedupe)")
            else:
                db.insert_block(current_chapter_num, next_block_num(current_chapter_num), "paragraph", content)
                info(f"[✓] Saved paragraph (chapter={current_chapter_num}): {content[:120]}...")

        # ---------- advance input by consumed content ----------
        full_text = strip_consumed(full_text, content)

        chunk_num += 1

    info("[Done] processing complete.")

if __name__ == "__main__":
    main()
