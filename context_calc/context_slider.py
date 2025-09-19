# text_finder.py
import re
from difflib import SequenceMatcher

def normalize_for_search(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\r\n", "\n")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def strip_consumed(full_text: str, content: str, min_ratio: float = 0.9) -> str:
    """
    Remove everything up to and including the first match of 'content'
    inside 'full_text'. Accepts fuzzy matches >= min_ratio.
    If nothing is found, return the original text.
    """
    if not content:
        return full_text

    # 1. direct search
    idx = full_text.find(content)
    if idx != -1:
        return full_text[idx + len(content):]

    # 2. normalized direct search
    norm_content = normalize_for_search(content)
    norm_text = normalize_for_search(full_text)
    idx = norm_text.find(norm_content)
    if idx != -1:
        return full_text[idx + len(content):]

    # 3. fuzzy search
    if norm_content and len(norm_content) > 20:  # avoid tiny fragments
        # sliding window over the full_text
        window_size = len(norm_content)
        best_ratio = 0.0
        best_pos = -1
        for i in range(0, max(1, len(norm_text) - window_size), max(10, window_size // 10)):
            candidate = norm_text[i:i + window_size]
            ratio = SequenceMatcher(None, norm_content, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i
                if ratio >= min_ratio:
                    break
        if best_pos != -1 and best_ratio >= min_ratio:
            # approximate cut using original full_text slice
            return full_text[best_pos + len(content):]

    # fallback: nothing matched
    return full_text



if __name__ == "__main__":
    import sys
    from pathlib import Path

    FULL_PATH = Path("context_calc/full.txt")
    REMOVE_PATH = Path("context_calc/remove_this.txt")
    OUT_PATH = Path("context_calc/remaining.txt")

    if not FULL_PATH.exists():
        print(f"[error] Missing {FULL_PATH.resolve()}")
        sys.exit(1)
    if not REMOVE_PATH.exists():
        print(f"[error] Missing {REMOVE_PATH.resolve()}")
        sys.exit(1)

    full_text = FULL_PATH.read_text(encoding="utf-8")
    to_remove = REMOVE_PATH.read_text(encoding="utf-8")

    before_len = len(full_text)
    result = strip_consumed(full_text, to_remove, min_ratio=0.90)
    after_len = len(result)

    OUT_PATH.write_text(result, encoding="utf-8")

    removed = before_len - after_len
    print(f"[ok] Wrote {OUT_PATH.resolve()}")
    print(f"[stats] original={before_len} chars, remaining={after_len} chars, removed≈{removed} chars")
