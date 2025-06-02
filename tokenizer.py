from flask import Flask, request, Response
import tiktoken
import json
import unicodedata
import re

app = Flask(__name__)

def clean_text(text: str) -> str:
    """
    Очищает входной текст от артефактов:
    1) Убирает BOM (U+FEFF) в начале строки, если он есть.
    2) Приводит к строке Python и заменяет '�' (U+FFFD) сразу на пробел.
    3) Удаляет все символы категории 'C*' (Control) и 'So' (Symbol, Other), заменяя их на пробел.
    4) Заменяет 'null' (в любом регистре) на пробел.
    5) Сводит несколько пробелов подряд к одному.
    """
    # 1) Если есть BOM (Byte Order Mark) в начале, убираем его
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2) Явно меняем символ Replacement Character '�' (U+FFFD) на пробел
    text = text.replace("\ufffd", " ")

    # 3) Пробегаем по всем символам, смотрим их Unicode-категории.
    #    - Control-символы (категория 'C*') и Replacement (U+FFFD, категория 'So') → ставим вместо них пробел.
    #    - Всё остальное оставляем «как есть».
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        # Если символ — любая категория, начинающаяся на 'C' (Control), 
        # или точная категория 'So' (Symbol, Other), то считаем его «артефактом»
        if cat.startswith("C") or cat == "So":
            cleaned_chars.append(" ")
        else:
            cleaned_chars.append(ch)
    text = "".join(cleaned_chars)

    # 4) Заменяем 'null' в любом регистре на пробел
    #    Например, 'null', 'NULL', 'Null' → ' '
    text = re.sub(r"null", " ", text, flags=re.IGNORECASE)

    # 5) Делаем цикл, чтобы свести подряд идущие пробелы к одному
    #    Например, 'Привет   мир' → 'Привет мир'
    while "  " in text:
        text = text.replace("  ", " ")

    return text.strip()


def pick_chunk_params(text: str, encoding_name: str = "cl100k_base") -> tuple[int, int]:
    """
    Автоподбор параметров chunk_size и overlap исходя из длины текста в токенах.
    Возвращает кортеж: (chunk_size, overlap).
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
    Разбивает текст на чанки по chunk_size токенов c перекрытием overlap токенов.
    Возвращает список декодированных строк (каждый элемент — chunk в текстовом виде).
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks: list[str] = []

    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        try:
            decoded = encoding.decode(chunk_tokens)
        except Exception:
            decoded = ""
        chunks.append(decoded)

    return chunks


@app.route('/split', methods=['POST'])
def split() -> Response:
    """
    HTTP POST /split
    Принимает JSON: { "text": "<любой текст>" }
    Возвращает JSON: { "chunks": [ "<чанк1>", "<чанк2>", ... ] }
    """
    # 1) Пробуем распарсить JSON
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

    # 2) Убедимся, что получили словарь
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

    # 4) Очищаем текст от всех артефактов
    text = clean_text(str(raw_text))

    # 5) Динамически подбираем chunk_size и overlap
    chunk_size, overlap = pick_chunk_params(text)

    # 6) Разбиваем очищённый текст на чанки
    chunks = split_chunks_by_tokens(text, chunk_size, overlap)

    # 7) Возвращаем JSON-ответ
    response_body = json.dumps({"chunks": chunks}, ensure_ascii=False)
    return Response(response_body, content_type="application/json")


if __name__ == "__main__":
    # Запуск локального Flask-сервера для отладки
    app.run(host="0.0.0.0", port=5555, debug=True)
