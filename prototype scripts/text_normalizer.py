#!/usr/bin/env python3
"""
text_normalizer_full.py

Block-level text normalizer using local Mistral inference runtime.

Features:
- Sliding-window extraction with overlap.
- Prompts model to output ONLY JSON {"blocks":[{"type":"header","content":...}, ...]}
- Robust matching of model-normalized content back into original text so cursor advances
  and we avoid infinite loops / duplicate extraction.
- SQLite DB storage for chapters + paragraphs.
- Saves failed raw outputs for debugging.

Configure via environment variables or edit the CONFIG section below.
"""

import os
import re
import json
import time
import traceback
import sqlite3
from pathlib import Path
from string import Template

# Local mistral runtime imports
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# --------------------- CONFIG ---------------------
MISTRAL_MODELS_PATH = Path(os.environ.get("MISTRAL_MODELS_PATH", "/home/rexford/models/Mistral-7B-Instruct-v0.3"))
INPUT_FILE = Path(os.environ.get("INPUT_FILE", "nameofthewind.txt"))
DB_FILE = Path(os.environ.get("DB_FILE", "paragraphs_mistral.db"))

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 1600))
OVERLAP = int(os.environ.get("OVERLAP", 300))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", 2))
MAX_GENERATION_TOKENS = int(os.environ.get("MAX_GENERATION_TOKENS", 512))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.0))
SAVE_FAILED_DIR = Path(os.environ.get("SAVE_FAILED_DIR", "failed_chunks"))
SAVE_FAILED_DIR.mkdir(exist_ok=True)

VERBOSE = True  # set False to reduce printing
# --------------------------------------------------

# Prompt template (Template with ${chunk} placeholder)
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

# --------------------- Helpers ---------------------
def info(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs, flush=True)

def find_tokenizer_file(model_path: Path):
    # typical names
    for name in ("tokenizer.model.v3", "tokenizer.model.v2", "tokenizer.model", "tokenizer.json"):
        p = model_path / name
        if p.exists():
            return p
    # fallback any starting with tokenizer
    for p in sorted(model_path.glob("tokenizer*")):
        if p.is_file():
            return p
    return None

def safe_save_failed(raw: str, chunk_num: int):
    fname = SAVE_FAILED_DIR / f"failed_chunk_{chunk_num}_{int(time.time())}.txt"
    with open(fname, "w", encoding="utf-8") as fh:
        fh.write(raw)
    return fname

def extract_first_json_with_key(text: str, required_key="blocks"):
    """
    Balanced-braces scan: find first JSON object that loads and contains required_key.
    Returns Python object or None.
    """
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
    """Collapse whitespace, normalize quotes, trim."""
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def find_in_full_text(content: str, full_text: str, cursor_pos: int) -> int:
    """
    Try to find content in full_text at or after cursor_pos.
    Returns index AFTER content (new cursor pos) or -1 if not found.
    """
    if not content:
        return -1

    # 1) exact match after cursor
    idx = full_text.find(content, cursor_pos)
    if idx != -1:
        return idx + len(content)

    # 2) normalized match
    norm_content = normalize_for_search(content)
    if norm_content:
        # create a normalized window of the suffix to search
        suffix_norm = normalize_for_search(full_text[cursor_pos:])
        pos_norm = suffix_norm.find(norm_content)
        if pos_norm != -1:
            # Try to map back using the first few characters of norm_content
            probe = norm_content[:120]
            if probe:
                m = re.search(re.escape(probe), full_text[cursor_pos:], flags=re.IGNORECASE)
                if m:
                    # return end of probe occurrence (approximate)
                    return cursor_pos + m.start() + len(probe)
            # fallback: approximate location using cursor_pos + pos_norm
            return cursor_pos + pos_norm + len(norm_content)

    # 3) prefix heuristics
    for L in (120, 80, 60, 40, 20):
        pref = content[:L].strip()
        if not pref:
            continue
        idx = full_text.find(pref, cursor_pos)
        if idx != -1:
            return idx + len(pref)
        pref_norm = normalize_for_search(pref)
        if pref_norm:
            idx = normalize_for_search(full_text[cursor_pos:]).find(pref_norm)
            if idx != -1:
                return cursor_pos + idx + len(pref_norm)

    return -1

# --------------------- Model init ---------------------
info("[info] model path:", MISTRAL_MODELS_PATH)
if not MISTRAL_MODELS_PATH.exists():
    raise FileNotFoundError(f"Model path not found: {MISTRAL_MODELS_PATH}")

tk_file = find_tokenizer_file(MISTRAL_MODELS_PATH)
if tk_file is None:
    files = "\n".join(sorted([p.name for p in MISTRAL_MODELS_PATH.iterdir() if p.is_file()]))
    raise FileNotFoundError(f"Tokenizer not found in {MISTRAL_MODELS_PATH}. Present files:\n{files}")

info("[info] tokenizer file:", tk_file)
tokenizer = MistralTokenizer.from_file(str(tk_file))
model = Transformer.from_folder(str(MISTRAL_MODELS_PATH))
info("[info] model & tokenizer loaded.")

# best-effort eos_id
try:
    eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
except Exception:
    eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None) or 0
info("[debug] eos_id:", eos_id)

def mistral_generate_text(prompt: str, max_new_tokens: int = MAX_GENERATION_TOKENS, temperature: float = TEMPERATURE):
    """Run the local Mistral generate and return decoded string."""
    # Build chat request
    request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    enc = tokenizer.encode_chat_completion(request)
    input_tokens = enc.tokens
    out_tokens_list, info_dict = generate([input_tokens], model, max_tokens=max_new_tokens, temperature=temperature, eos_id=eos_id)
    # decode first output
    try:
        decoded = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens_list[0])
    except Exception:
        try:
            decoded = tokenizer.decode(out_tokens_list[0])
        except Exception:
            decoded = "[ERROR decoding tokens]"
    return decoded

# --------------------- DB init ---------------------
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS chapters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_num INTEGER,
    header TEXT
);
""")
cur.execute("""
CREATE TABLE IF NOT EXISTS paragraphs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id INTEGER,
    paragraph_num INTEGER,
    content TEXT,
    FOREIGN KEY(chapter_id) REFERENCES chapters(id)
);
""")
conn.commit()

def db_has_paragraph(content: str) -> bool:
    cur.execute("SELECT 1 FROM paragraphs WHERE content = ? LIMIT 1", (content,))
    return cur.fetchone() is not None

# --------------------- Load input ---------------------
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

full_text = INPUT_FILE.read_text(encoding="utf-8")
# basic clean
full_text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", full_text)
full_text = re.sub(r"\r\n", "\n", full_text)

total_len = len(full_text)
info(f"[info] input length {total_len} chars")

cursor = 0
chunk_num = 1
chapter_counter = 0
current_chapter_id = None
paragraph_counters = {}

# --------------------- Main sliding window loop ---------------------
while cursor < len(full_text):
    window = full_text[cursor: cursor + WINDOW_SIZE]
    prompt = PROMPT_TEMPLATE.safe_substitute(chunk=window)

    parsed = None
    raw_out = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            info(f"[chunk {chunk_num}] generating (attempt {attempt}) prompt len={len(prompt)}")
            raw_out = mistral_generate_text(prompt, max_new_tokens=MAX_GENERATION_TOKENS, temperature=TEMPERATURE)
            info(f"[chunk {chunk_num}] raw output preview:\n{raw_out[:1000].replace(chr(10), ' ')}\n")
            parsed = extract_first_json_with_key(raw_out, required_key="blocks")
            if parsed is not None:
                break
            else:
                info(f"[warn] JSON extraction failed on chunk {chunk_num}, attempt {attempt}")
                fname = safe_save_failed(raw_out, chunk_num)
                info("[debug] saved failed raw to", fname)
        except Exception as e:
            info(f"[error] generation exception on chunk {chunk_num}: {e}")
            traceback.print_exc()
            fname = safe_save_failed(f"EXC:{e}\n{traceback.format_exc()}", chunk_num)
            info("[debug] saved exception to", fname)
        time.sleep(0.25)

    if parsed is None:
        info(f"[x] skipping chunk {chunk_num} — no valid JSON after {MAX_RETRIES} attempts")
        cursor += max(1, WINDOW_SIZE - OVERLAP)
        chunk_num += 1
        continue

    blocks = parsed.get("blocks", [])
    if not isinstance(blocks, list):
        info(f"[warn] 'blocks' not a list for chunk {chunk_num}, skipping")
        cursor += max(1, WINDOW_SIZE - OVERLAP)
        chunk_num += 1
        continue

    # handle blocks & compute advancement
    used_advance = 0
    made_progress = False

    for block in blocks:
        btype = (block.get("type") or "").lower()
        content = (block.get("content") or "").strip()
        if not content:
            continue

        # dedupe paragraphs
        if btype == "paragraph" and db_has_paragraph(content):
            info(f"[skip] paragraph already in DB (dedupe): {content[:60]}...")
            adv_pos = find_in_full_text(content, full_text, cursor)
            if adv_pos != -1:
                used_advance = max(used_advance, adv_pos - cursor)
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
            # fallback treat unknown block types as paragraph
            if current_chapter_id is None:
                chapter_counter += 1
                cur.execute("INSERT INTO chapters (chapter_num, header) VALUES (?, ?)", (0, "Prologue/Intro"))
                conn.commit()
                current_chapter_id = cur.lastrowid
                paragraph_counters[current_chapter_id] = 1
            pnum = paragraph_counters[current_chapter_id]
            cur.execute(
                "INSERT INTO paragraphs (chapter_id, paragraph_num, content) VALUES (?, ?, ?)",
                (current_chapter_id, pnum, content)
            )
            conn.commit()
            paragraph_counters[current_chapter_id] += 1
            info(f"[?] Saved unknown block as paragraph {pnum} (chapter_id={current_chapter_id})")

        # try to find the content forward of cursor
        adv_pos = find_in_full_text(content, full_text, cursor)
        if adv_pos != -1:
            adv = adv_pos - cursor
            used_advance = max(used_advance, adv)
            made_progress = True
        else:
            # if not found but exists in full_text somewhere, remove first occurrence to break loops
            if content in full_text:
                info("[fixup] content present elsewhere but not after cursor: removing first occurrence to avoid loop")
                full_text = full_text.replace(content, "", 1)
                made_progress = True
            else:
                # fallback: schedule default large advance to avoid stuck
                fallback_adv = max(len(content), WINDOW_SIZE - OVERLAP)
                info(f"[warn] couldn't locate content in text; will fallback-advance {fallback_adv} chars")
                used_advance = max(used_advance, fallback_adv)
                made_progress = True

    # advance cursor
    if made_progress and used_advance and used_advance > 10:
        cursor += used_advance
        info(f"[info] advanced cursor by matched content: {used_advance} chars -> cursor={cursor}")
    else:
        step = max(1, WINDOW_SIZE - OVERLAP)
        cursor += step
        info(f"[info] advanced cursor by default step: {step} chars -> cursor={cursor}")

    # clamp cursor
    if cursor < 0:
        cursor = 0
    if cursor > len(full_text):
        cursor = len(full_text)

    chunk_num += 1

info("[Done] processing complete.")
conn.close()
