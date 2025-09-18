# mistral_interface.py
"""
Small wrapper around local mistral_inference runtime.

Provides:
- load_mistral(model_path) -> tokenizer, model, eos_id
- generate_text(tokenizer, model, prompt, max_new_tokens, temperature) -> decoded string
"""

from pathlib import Path
import traceback

# local mistral imports
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

def find_tokenizer_file(model_path: Path):
    for name in ("tokenizer.model.v3", "tokenizer.model.v2", "tokenizer.model", "tokenizer.json"):
        p = model_path / name
        if p.exists():
            return p
    for p in sorted(model_path.glob("tokenizer*")):
        if p.is_file():
            return p
    return None

def load_mistral(model_path):
    """
    Load tokenizer and model from model_path (Path or str).
    Returns: (tokenizer, model, eos_id)
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    tk_file = find_tokenizer_file(model_path)
    if tk_file is None:
        files = "\n".join(sorted([p.name for p in model_path.iterdir() if p.is_file()]))
        raise FileNotFoundError(f"Tokenizer not found in {model_path}. Present files:\n{files}")

    tokenizer = MistralTokenizer.from_file(str(tk_file))
    model = Transformer.from_folder(str(model_path))

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
    decoded = None
    try:
        decoded = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens_list[0])
    except Exception:
        try:
            decoded = tokenizer.decode(out_tokens_list[0])
        except Exception:
            decoded = "[ERROR decoding tokens]"
    return decoded
