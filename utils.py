import re

#################
# STRINGS UTILS #
#################

PUNCTUATIONS = ["...", ",", ".", "!", "?", ":"]
PUNCTS_STR = "[" + ", ".join(f"'{p}'" for p in PUNCTUATIONS) + "]"

# Слаги для колонок CSV (используется в bench.py)
PUNCT_SLUG = {
    "...": "ellipsis",
    ",": "comma",
    ".": "dot",
    "!": "excl",
    "?": "quest",
    ":": "colon",
}

OUT_RE = re.compile(r"<out>(.*?)</out>", flags=re.DOTALL | re.IGNORECASE)

def extract_out_payload(text: str) -> str:
    m = OUT_RE.search(text)
    return m.group(1).strip() if m else text.strip()

#############
# RUN UTILS #
#############

def iter_batches(items: list[dict[str, str]], bs: int):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]
