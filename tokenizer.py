from flask import Flask, request, Response
import tiktoken
import json
import unicodedata
import re

app = Flask(__name__)

def clean_text(text: str) -> str:
    """
    Очищает входной текст от артефактов, сохраняя символы новой строки:
    1) Убирает BOM (U+FEFF) в начале строки, если есть.
    2) Заменяет символ Replacement Character '�' (U+FFFD) на пробел.
    3) Удаляет все символы категории 'C*' (Control, кроме '\n') и 'So' (Symbol, Other), заменяя их на пробел.
    4) Заменяет 'null' (в любом регистре) на пробел.
    5) Сводит несколько пробелов подряд к одному.
    6) Нормализует CRLF → LF (Windows → Unix) и сводит больше двух подряд '\n' к двойному '\n\n'.
    """
    # 1) Убираем BOM в начале
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2) Заменяем Replacement Character '�' (U+FFFD) на пробел
    text = text.replace("\ufffd", " ")

    # 3) Пробегаем по всем символам.
    #    - Сохраняем '\n' (LF), остальные \r,\t и т.п. будем удалять/заменять.
    #    - Убираем символы категории 'So' (Symbol, Other) и 'C*' (Control), кроме '\n'.
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if ch == "\n":
            cleaned_chars.append(ch)
        elif cat.startswith("C") or cat == "So":
            # Заменяем управляющие символы (кроме '\n') и символы категории 'So' на пробел
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # 4) Заменяем 'null' (в любом регистре) на пробел
    text = re.sub(r"null", " ", text, flags=re.IGNORECASE)

    # 5) Сводим несколько пробелов подряд к одному
    while "  " in text:
        text = text.replace("  ", " ")

    # 6) Нормализуем переводы строк:
    #    - Сначала заменим CRLF (\r\n) и одиночный CR (\r) на одиночный LF (\n)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    #    - Если получилось более двух '\n' подряд, сводим к '\n\n' (максимум двойной разрыв строки)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Убираем пробелы в начале/конце всего текста
    return text.strip()

def pick_chunk_params(text: str, encoding_name: str = "cl100k_base") -> tuple[int, int]:
    """
    Простой подбор параметров chunk_size и overlap по длине текста (в токенах).
    Возвращает (chunk_size, overlap).
    """
    encoding = tiktoken.get_encoding(encoding_name)
    n_tokens = len(encoding.encode(text))
    if n_tokens < 300:
        return n_tokens, 0
    if n_tokens < 2000:
        return 512, 64
    if n_tokens < 5000:
        return 1024, 128
    if n_tokens < 12000:
        return 1500, 200
    return 2000, 250

def split_chunks_by_tokens(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбивает текст на чанки по chunk_size токенов c overlap (перекрытием токенов).
    В данном примере мы просто нарезаем последовательно по step=chunk_size-overlap,
    сохраняя в чанке тексты, в которых есть символы '\n'.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks: list[str] = []

    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(tokens), step):
        sub_tokens = tokens[i : i + chunk_size]
        try:
            decoded = encoding.decode(sub_tokens)
        except Exception:
            decoded = ""
        # В decoded уже есть '\n'. JSON-сериализатор сам преобразует их в '\\n'
        chunks.append(decoded)

    return chunks

@app.route('/split', methods=['POST'])
def split() -> Response:
    """
    HTTP POST /split
    Вход: JSON {"text": "<любой длинный текст>"}
    Выход: JSON [{"chanks": [ "<чанк1>", "<чанк2>", ... ]}]
    """
    # 1) Парсим JSON
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return Response(
            json.dumps(
                {"error": "Невозможно распарсить JSON", "exception": str(e)},
                ensure_ascii=False,
            ),
            status=400,
            content_type="application/json",
        )

    # 2) Убеждаемся, что data — это словарь
    if not isinstance(data, dict):
        return Response(
            json.dumps({"error": "Ожидался JSON-объект"}, ensure_ascii=False),
            status=400,
            content_type="application/json",
        )

    # 3) Берём поле "text"
    raw_text = data.get("text")
    if raw_text is None:
        return Response(
            json.dumps(
                {"error": "В теле запроса отсутствует ключ 'text'"},
                ensure_ascii=False,
            ),
            status=400,
            content_type="application/json",
        )

    # 4) Очищаем текст от артефактов, сохраняя '\n'
    text = clean_text(str(raw_text))

    # 5) Подбираем параметры chunk_size, overlap
    chunk_size, overlap = pick_chunk_params(text)

    # 6) Разбиваем на чанки (список строк с '\n')
    chunks = split_chunks_by_tokens(text, chunk_size, overlap)

    # 7) Формируем ответ именно в том формате, который вы хотите:
    #    список, в котором один объект с ключом "chanks" и массивом строк
    out = [{"chanks": chunks}]

    # При сериализации JSON-строки '\n' превратятся в '\\n', и вы получите нужный вид
    response_body = json.dumps(out, ensure_ascii=False)
    return Response(response_body, content_type="application/json")


if __name__ == "__main__":
    # Локальный запуск Flask для отладки
    app.run(host="0.0.0.0", port=5555, debug=True)
