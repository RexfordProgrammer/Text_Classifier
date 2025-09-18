# minimal_mistral_io_fixed.py
"""
Minimal Mistral local inference I/O (fixed).
 - Reads prompt from ./prompt.txt
 - Writes model output to ./response.txt (full raw string)
 - Model path: either first CLI arg or MISTRAL_MODELS_PATH env or fallback variable below.
"""

import os
import sys
import traceback
from pathlib import Path

from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# ---- Config ----
# You can override by passing model folder as first CLI arg,
# or set MISTRAL_MODELS_PATH in env.
DEFAULT_MODEL_PATH = "/home/rexford/models/Mistral-7B-Instruct-v0.3"
MODEL_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get("MISTRAL_MODELS_PATH", DEFAULT_MODEL_PATH))

PROMPT_FILE = Path("prompt.txt")
OUTPUT_FILE = Path("response.txt")
PROMPT_COPY = Path("prompt_sent.txt")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 512))
TEMPERATURE = float(os.environ.get("TEMP", 0.0))

# ---- Helpers ----
def find_tokenizer_file(model_path: Path):
    for name in ("tokenizer.model.v3", "tokenizer.model.v2", "tokenizer.model", "tokenizer.json"):
        p = model_path / name
        if p.exists():
            return p
    # fallback: any file starting with tokenizer
    for p in sorted(model_path.glob("tokenizer*")):
        if p.is_file():
            return p
    return None

def safe_decode(tokenizer, out_tokens):
    # Try the common decode entry points used by Mistral tooling.
    # Return a string (or raise).
    try:
        return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens)
    except Exception:
        pass
    try:
        # some tokenizers expose top-level decode
        return tokenizer.decode(out_tokens)
    except Exception:
        pass
    # last ditch: join ints as string (for debugging)
    return "ERROR: could not decode tokens with tokenizer API. tokens: " + " ".join(map(str, out_tokens[:200]))

# ---- Main ----
def main():
    if not MODEL_PATH.exists():
        print(f"[error] model folder does not exist: {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)

    if not PROMPT_FILE.exists():
        print(f"[error] prompt file not found: {PROMPT_FILE}", file=sys.stderr)
        sys.exit(1)

    prompt_text = PROMPT_FILE.read_text(encoding="utf-8")
    PROMPT_COPY.write_text(prompt_text, encoding="utf-8")  # preserve a copy of the prompt used

    print(f"[info] Read prompt ({len(prompt_text)} chars). Using model folder: {MODEL_PATH}")
    print(f"[info] Prompt preview (first 1000 chars):\n{prompt_text[:1000]!r}\n--- end preview ---")

    tk_file = find_tokenizer_file(MODEL_PATH)
    if not tk_file:
        files = "\n".join(sorted([p.name for p in MODEL_PATH.iterdir() if p.is_file()]))
        print(f"[error] tokenizer file not found in {MODEL_PATH}. Files:\n{files}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"[info] Loading tokenizer from {tk_file} ...")
        tokenizer = MistralTokenizer.from_file(str(tk_file))
        print("[info] Loading model (this may take some time) ...")
        model = Transformer.from_folder(str(MODEL_PATH))
    except Exception as e:
        print("[error] failed to load tokenizer/model:", e, file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # eos id detection (best-effort)
    try:
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
    except Exception:
        eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None) or 0

    print(f"[info] Tokenizer file: {tk_file.name}  eos_id={eos_id}")
    print(f"[info] Generating (max_new_tokens={MAX_NEW_TOKENS}, temp={TEMPERATURE})...")

    try:
        request = ChatCompletionRequest(messages=[UserMessage(content=prompt_text)])
        enc = tokenizer.encode_chat_completion(request)
        input_tokens = enc.tokens

        out_tokens_list, gen_info = generate([input_tokens], model, max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, eos_id=eos_id)

        if not out_tokens_list:
            raise RuntimeError("generate returned no tokens")

        decoded = safe_decode(tokenizer, out_tokens_list[0])

        # Save decoded output
        OUTPUT_FILE.write_text(decoded, encoding="utf-8")
        print(f"[ok] Model output written to {OUTPUT_FILE} ({len(decoded)} chars).")
        # optional: print a short preview
        print("\n[preview] first 1000 chars of model output:\n")
        print(decoded[:1000])
        print("\n[done preview]\n")
    except Exception as e:
        print("[error] generation failed:", e, file=sys.stderr)
        traceback.print_exc()
        # save exception dump
        OUTPUT_FILE.write_text(f"EXCEPTION: {e}\n\n{traceback.format_exc()}", encoding="utf-8")
        sys.exit(2)


if __name__ == "__main__":
    main()
