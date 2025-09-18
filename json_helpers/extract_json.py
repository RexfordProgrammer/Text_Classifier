import json
import re


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