import os
import re
import json
import time
import traceback
import sqlite3
from pathlib import Path
from string import Template

from mistral.mistral_interface import load_mistral, generate_text

# ---------- CONFIG ----------
MISTRAL_MODELS_PATH = Path(os.environ.get("MISTRAL_MODELS_PATH", "/home/rexford/models/Mistral-7B-Instruct-v0.3"))
INPUT_FILE = Path(os.environ.get("INPUT_FILE", "nameofthewind.txt"))
DB_FILE = Path(os.environ.get("DB_FILE", "paragraphs_mistral.db"))

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 1600))
OVERLAP = int(os.environ.get("OVERLAP", 300))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 2))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
SAVE_FAILED_DIR = Path(os.environ.get("SAVE_FAILED_DIR", "failed_chunks"))
SAVE_FAILED_DIR.mkdir(exist_ok=True)

VERBOSE = True

# prompt template (use Template to safely inject chunk)
PROMPT_TEMPLATE = Template("""You are a strict text normalizer.
Output ONLY valid JSON (no prose, no explanation) following this exact schema:

{
  "blocks": [
    { "type": "header", "content": "CHAPTER ONE: Title" },
    { "type": "paragraph", "content": "First paragraph text..." }
  ]
}

When given raw text, split it into logical blocks (header or paragraph) and return a single JSON
object containing "blocks" (an array). Each block must be {"type": "header"|"paragraph", "content": "..." }.

Now normalize this text and output valid JSON only:

""" + '"""' + "${chunk}" + '"""')

# ----------------- helpers -----------------
def info(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)

def safe_save_failed(raw: str, chunk_num: int):
    fname = SAVE_FAILED_DIR / f"failed_chunk_{chunk_num}_{int(time.time())}.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(raw)
    return fname

def extract_first_json_with_key(text: str, required_key="blocks"):
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in starts:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        obj = json.loads(candidate)
                        if required_key is None or required_key in obj:
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
    return None

def normalize_for_search(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def find_in_full_text(content: str, full_text: str, cursor_pos: int) -> int:
    if not content:
        return -1
    idx = full_text.find(content, cursor_pos)
    if idx != -1:
        return idx + len(content)
    # normalized
    norm_content = normalize_for_search(content)
    if norm_content:
        suffix_norm = normalize_for_search(full_text[cursor_pos:])
        pos_norm = suffix_norm.find(norm_content)
        if pos_norm != -1:
            # approximate mapping: try to locate first few chars
            probe = norm_content[:120]
            if probe:
                m = re.search(re.escape(probe), full_text[cursor_pos:], flags=re.IGNORECASE)
                if m:
                    return cursor_pos + m.start() + len(probe)
            return cursor_pos + pos_norm + len(norm_content)
    for L in (120, 80, 60, 40, 20):
        pref = content[:L].strip()
        if not pref:
            continue
        idx = full_text.find(pref, cursor_pos)
        if idx != -1:
            return idx + len(pref)
    return -1

def db_has_paragraph(cur, content: str) -> bool:
    cur.execute("SELECT 1 FROM paragraphs WHERE content = ? LIMIT 1", (content,))
    return cur.fetchone() is not None

# ----------------- main -----------------
def main():
    info("[info] loading model/tokenizer from:", MISTRAL_MODELS_PATH)
    tokenizer, model, eos_id = load_mistral(MISTRAL_MODELS_PATH)
    info("[info] model loaded. eos_id:", eos_id)

    # DB init
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chapters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chapter_num INTEGER,
        header TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS paragraphs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chapter_id INTEGER,
        paragraph_num INTEGER,
        content TEXT,
        FOREIGN KEY(chapter_id) REFERENCES chapters(id)
    )
    """)
    conn.commit()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")
    full_text = INPUT_FILE.read_text(encoding="utf-8")
    # small cleanup
    full_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", full_text)
    full_text = re.sub(r"\r\n", "\n", full_text)

    total_len = len(full_text)
    info(f"[info] input length {total_len} chars")

    cursor = 0
    chunk_num = 1
    chapter_counter = 0
    current_chapter_id = None
    paragraph_counters = {}

    while cursor < len(full_text):
        window = full_text[cursor: cursor + WINDOW_SIZE]
        prompt = PROMPT_TEMPLATE.safe_substitute(chunk=window)

        parsed = None
        raw_out = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                info(f"[chunk {chunk_num}] generating (attempt {attempt}) prompt len={len(prompt)}")
                raw_out = generate_text(tokenizer, model, eos_id, prompt, temperature=TEMPERATURE)
                info(f"[chunk {chunk_num}] raw preview:\n{raw_out[:800].replace(chr(10), ' ')}\n")
                parsed = extract_first_json_with_key(raw_out, required_key="blocks")
                if parsed is not None:
                    break
                else:
                    info(f"[warn] JSON extraction failed for chunk {chunk_num} attempt {attempt}")
                    fname = safe_save_failed(raw_out, chunk_num)
                    info("[debug] saved failed raw to", fname)
            except Exception as e:
                info(f"[error] generation exception on chunk {chunk_num}: {e}")
                traceback.print_exc()
                fname = safe_save_failed(f"EXC:{e}\n{traceback.format_exc()}", chunk_num)
                info("[debug] saved exception to", fname)
            time.sleep(0.25)

        if parsed is None:
            info(f"[x] skipping chunk {chunk_num} — no valid JSON")
            cursor += max(1, WINDOW_SIZE - OVERLAP)
            chunk_num += 1
            continue

        blocks = parsed.get("blocks", [])
        if not isinstance(blocks, list):
            info(f"[warn] 'blocks' not a list; skipping chunk {chunk_num}")
            cursor += max(1, WINDOW_SIZE - OVERLAP)
            chunk_num += 1
            continue

        used_advance = 0
        made_progress = False

        for block in blocks:
            btype = (block.get("type") or "").lower()
            content = (block.get("content") or "").strip()
            if not content:
                continue

            if btype == "paragraph" and db_has_paragraph(cur, content):
                info("[skip] paragraph already exists in DB (dedupe)")
                adv = find_in_full_text(content, full_text, cursor)
                if adv != -1:
                    used_advance = max(used_advance, adv - cursor)
                    made_progress = True
                continue

            if btype == "header":
                chapter_counter += 1
                cur.execute("INSERT INTO chapters (chapter_num, header) VALUES (?, ?)", (chapter_counter, content))
                conn.commit()
                current_chapter_id = cur.lastrowid
                paragraph_counters[current_chapter_id] = 1
                info(f"[+] Chapter {chapter_counter}: {content}")

            elif btype == "paragraph":
                if current_chapter_id is None:
                    chapter_counter += 1
                    cur.execute("INSERT INTO chapters (chapter_num, header) VALUES (?, ?)", (0, "Prologue/Intro"))
                    conn.commit()
                    current_chapter_id = cur.lastrowid
                    paragraph_counters[current_chapter_id] = 1
                    info(f"[!] Inserted Prologue/Intro as chapter_id={current_chapter_id}")

                pnum = paragraph_counters[current_chapter_id]
                cur.execute(
                    "INSERT INTO paragraphs (chapter_id, paragraph_num, content) VALUES (?, ?, ?)",
                    (current_chapter_id, pnum, content)
                )
                conn.commit()
                paragraph_counters[current_chapter_id] += 1
                info(f"[✓] Saved paragraph {pnum} (chapter_id={current_chapter_id}): {content[:120]}...")

            else:
                # fallback as paragraph
                if current_chapter_id is None:
                    chapter_counter += 1
                    cur.execute("INSERT INTO chapters (chapter_num, header) VALUES (?, ?)", (0, "Prologue/Intro"))
                    conn.commit()
                    current_chapter_id = cur.lastrowid
                    paragraph_counters[current_chapter_id] = 1
                pnum = paragraph_counters[current_chapter_id]
                cur.execute("INSERT INTO paragraphs (chapter_id, paragraph_num, content) VALUES (?, ?, ?)",
                            (current_chapter_id, pnum, content))
                conn.commit()
                paragraph_counters[current_chapter_id] += 1
                info(f"[?] Saved unknown block as paragraph {pnum} (chapter_id={current_chapter_id})")

            adv_pos = find_in_full_text(content, full_text, cursor)
            if adv_pos != -1:
                used_advance = max(used_advance, adv_pos - cursor)
                made_progress = True
            else:
                if content in full_text:
                    info("[fixup] content appears earlier/later in text; removing first occurrence to avoid loop")
                    full_text = full_text.replace(content, "", 1)
                    made_progress = True
                else:
                    fallback = max(len(content), WINDOW_SIZE - OVERLAP)
                    used_advance = max(used_advance, fallback)
                    made_progress = True

        if made_progress and used_advance and used_advance > 10:
            cursor += used_advance
            info(f"[info] advanced cursor by matched content: {used_advance} -> cursor={cursor}")
        else:
            step = max(1, WINDOW_SIZE - OVERLAP)
            cursor += step
            info(f"[info] advanced cursor by default step: {step} -> cursor={cursor}")

        if cursor < 0:
            cursor = 0
        if cursor > len(full_text):
            cursor = len(full_text)

        chunk_num += 1

    info("[Done] processing complete.")
    conn.close()

if __name__ == "__main__":
    main()
