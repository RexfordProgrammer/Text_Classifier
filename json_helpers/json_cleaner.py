#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Tuple, List

# clean_json_file("json_helpers/local.json","json_helpers/cleaned_local.json", "log.txt")
# normalize_quotes(raw) output is cleaned json

SMART_TO_STRAIGHT = {
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
}

def normalize_quotes(s: str) -> str:
    for k, v in SMART_TO_STRAIGHT.items():
        s = s.replace(k, v)
    return s

def try_load(text: str):
    return json.loads(text)

def _repair_content_strings(text: str, changes: List[str]) -> str:
    """
    Scan the JSON text and, for every `"content": "<...>"`, ensure any inner
    unescaped quotes are escaped so the overall JSON remains valid.

    """
    out = []
    i = 0
    n = len(text)

    # helper
    def peek_nonspace(j: int) -> str:
        while j < n and text[j].isspace():
            j += 1
        return text[j] if j < n else ""

    while i < n:
        if text.startswith('"content"', i):
            # copy '"content"'
            out.append('"content"')
            i += len('"content"')
            # skip spaces
            while i < n and text[i].isspace():
                out.append(text[i]); i += 1
            # expect :
            if i < n and text[i] == ':':
                out.append(':'); i += 1
            # skip spaces
            while i < n and text[i].isspace():
                out.append(text[i]); i += 1
            # expect opening quote for the value
            if i < n and text[i] == '"':
                out.append('"'); i += 1
                buf = []
                escaped = False
                start_pos = i
                while i < n:
                    ch = text[i]
                    if escaped:
                        buf.append(ch)
                        escaped = False
                        i += 1
                        continue

                    if ch == '\\':
                        buf.append(ch)
                        escaped = True
                        i += 1
                        continue

                    if ch == '"':
                        # possible string end; check what follows
                        nxt = peek_nonspace(i + 1)
                        if nxt in (',', '}', ']'):
                            # real end of content string
                            out.append(''.join(buf))
                            out.append('"')
                            i += 1
                            break
                        else:
                            # inner quote — escape it
                            buf.append('\\"')
                            changes.append(f'Escaped inner quote in "content" at approx index {i}')
                            i += 1
                            continue

                    # normal char
                    buf.append(ch)
                    i += 1

                continue  # continue main loop after handling this "content" value

            else:
                # didn't find expected quote; just continue normally
                continue
        else:
            out.append(text[i])
            i += 1

    return ''.join(out)

def clean_json_file(input_path: str, output_path: str, log_path: str) -> None:
    inp = Path(input_path)
    outp = Path(output_path)
    logp = Path(log_path)

    raw = inp.read_text(encoding="utf-8")

    # 1) try as-is
    try:
        try_load(raw)
        outp.write_text(raw, encoding="utf-8")
        logp.write_text("[ok] Input was already valid JSON. No changes made.\n", encoding="utf-8")
        return
    except json.JSONDecodeError as e:
        # proceed with fixes
        pass

    changes: List[str] = []

    # 2) normalize curly quotes to straight
    normalized = normalize_quotes(raw)
    if normalized != raw:
        changes.append("Normalized curly quotes to ASCII quotes.")

    # 3) repair inner quotes inside "content"
    repaired = _repair_content_strings(normalized, changes)

    # 4) try to load after repair
    try:
        obj = try_load(repaired)
        # success: pretty-print the cleaned JSON
        pretty = json.dumps(obj, ensure_ascii=False, indent=2)
        outp.write_text(pretty, encoding="utf-8")
        # write log
        with logp.open("w", encoding="utf-8") as lf:
            if not changes:
                lf.write("[warn] No specific changes recorded, but initial parse failed and repaired text now parses.\n")
            else:
                lf.write("\n".join(changes) + "\n")
            lf.write("[done] JSON parsed successfully after repair.\n")
    except json.JSONDecodeError as e:
        # still bad; save best-effort repaired text and error
        outp.write_text(repaired, encoding="utf-8")
        with logp.open("w", encoding="utf-8") as lf:
            lf.write("\n".join(changes) + "\n")
            lf.write(f"[fail] Still invalid after repair: {e}\n")

if __name__ == "__main__":
    clean_json_file("json_helpers/local.json","json_helpers/cleaned_local.json", "log.txt")
