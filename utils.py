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

# --- Подготовка регулярного выражения ---
# Разбор паттерна r'<(\w+)>(.*?)</\1>'
# <       - начало открывающего тега
# (\w+)   - Группа 1: Имя тега (буквы, цифры, подчеркивание). Мы запоминаем это.
# >       - конец открывающего тега
# (.*?)   - Группа 2: Содержимое (нежадный поиск)
# </      - начало закрывающего тега
# \1      - ОБРАТНАЯ ССЫЛКА: Здесь должно быть то же самое слово, что нашлось в Группе 1
# >       - конец закрывающего тега
ANY_TAG_PATTERN = r'<(\w+)>(.*?)</\1>'


def extract_content(source_text):
    matches = re.findall(ANY_TAG_PATTERN, source_text)
    # Нам нужен только второй элемент кортежа (содержимое)
    return [match[1] for match in matches]

# --- Точная замена с сохранением структуры ---
def inject_content_safe(source_text: str, new_values: list[str]) -> str:
    value_iter = iter(new_values)
    
    def replacer(match):
        # match.group(1) - это имя тега (например, "tag")
        # match.group(2) - это старый текст (нам он не нужен для замены, но он есть)
        
        tag_name = match.group(1)
        try:
            new_content = next(value_iter)
            # Реконструируем строку: <имя>новое_содержимое</имя>
            return f"<{tag_name}>{new_content}</{tag_name}>"
        
        except StopIteration:
            # Если данные кончились, возвращаем как было
            return match.group(0)

    return re.sub(ANY_TAG_PATTERN, replacer, source_text)

#############
# RUN UTILS #
#############

def iter_batches(items: list[dict[str, str]], bs: int):
    for i in range(0, len(items), bs):
        yield items[i:i + bs]
