# mistral_interface.py
"""
Small wrapper around local mistral_inference runtime.

Provides:
- load_mistral() -> tokenizer, model, eos_id
- generate_text(tokenizer, model, prompt, max_new_tokens, temperature) -> decoded string
"""

from pathlib import Path

# local mistral imports
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

# Global model path (constant)
MODEL_PATH = Path("/home/rexford/models/Mistral-7B-Instruct-v0.3")

def find_tokenizer_file(model_path: Path):
    for name in ("tokenizer.model.v3", "tokenizer.model.v2", "tokenizer.model", "tokenizer.json"):
        p = model_path / name
        if p.exists():
            return p
    for p in sorted(model_path.glob("tokenizer*")):
        if p.is_file():
            return p
    return None

def load_mistral():
    """
    Load tokenizer and model from MODEL_PATH (global).
    Returns: (tokenizer, model, eos_id)
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model path not found: {MODEL_PATH}")

    tk_file = find_tokenizer_file(MODEL_PATH)
    if tk_file is None:
        files = "\n".join(sorted([p.name for p in MODEL_PATH.iterdir() if p.is_file()]))
        raise FileNotFoundError(f"Tokenizer not found in {MODEL_PATH}. Present files:\n{files}")

    tokenizer = MistralTokenizer.from_file(str(tk_file))
    model = Transformer.from_folder(str(MODEL_PATH))

    # best-effort eos_id
    try:
        eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
    except Exception:
        eos_id = getattr(tokenizer, "eos_id", None) or getattr(tokenizer, "eos_token_id", None) or 0

    return tokenizer, model, eos_id

def generate_text(tokenizer, model, eos_id, prompt: str, max_new_tokens: int = 2048, temperature: float = 0.0):
    """
    Generate text using the loaded tokenizer+model.
    prompt: full text prompt (chat format expected by tokenizer.encode_chat_completion).
    Returns decoded string.
    """
    # Build chat request (model uses the chat-style encoding in examples)
    request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
    enc = tokenizer.encode_chat_completion(request)
    input_tokens = enc.tokens

    # call local generate
    out_tokens_list, info = generate([input_tokens], model, max_tokens=max_new_tokens, temperature=temperature, eos_id=eos_id)

    # decode using the same tokenizer wrapper used elsewhere
    try:
        return tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens_list[0])
    except Exception:
        try:
            return tokenizer.decode(out_tokens_list[0])
        except Exception:
            return "[ERROR decoding tokens]"
