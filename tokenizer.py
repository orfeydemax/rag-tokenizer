from flask import Flask, request, Response
import tiktoken
import json
import unicodedata
import re

app = Flask(__name__)

def clean_text(text: str) -> str:
    """
    Очищает входной текст от артефактов:
    1) Убирает BOM (U+FEFF) в начале строки, если есть.
    2) Заменяет символ Replacement Character '�' (U+FFFD) на пробел.
    3) Удаляет любые символы с Unicode-категорией 'C' (Control, Format, Private Use, Surrogate и т.д.).
    4) Заменяет 'null' (в любом регистре) на пробел.
    5) Сводит несколько пробелов подряд к одному.
    """
    # 1) Если в начале стоит BOM (U+FEFF), убираем его
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2) Заменяем символ Replacement Character '�' (U+FFFD) на пробел
    text = text.replace("\ufffd", " ")

    # 3) Удаляем все символы, у которых Unicode-категория начинается на 'C'
    #    (такие символы — управляющие, форматирующие, неназначенные и т.д.)
    cleaned_chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if not cat.startswith("C"):
            cleaned_chars.append(ch)
        else:
            # Вместо этих символов просто ставим пробел, чтобы не сломать слова
            cleaned_chars.append(" ")
    text = "".join(cleaned_chars)

    # 4) Заменяем "null" в любом регистре на пробел
    #    re.IGNORECASE позволяет удалить 'null', 'NULL', 'Null' и т.д.
    text = re.sub(r"null", " ", text, flags=re.IGNORECASE)

    # 5) Сводим подряд идущие пробелы к одному
    #    Например, если получилось несколько пробелов из-за удаления символов категорий 'C'
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

    # Шаг: сколько смещаться дальше, чтобы сделать следующий chunk
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        try:
            decoded = encoding.decode(chunk_tokens)
        except Exception:
            # В случае ошибки декодирования возвращаем пустую строку
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
    # 1) Парсим JSON из тела запроса
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

    # 2) Проверяем, что JSON — это словарь
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
                {"error": "В теле запроса отсутствует ключ 'text'"}, ensure_ascii=False
            ),
            status=400,
            content_type="application/json",
        )

    # 4) Очищаем текст от артефактов
    text = clean_text(str(raw_text))

    # 5) Автоподбираем chunk_size и overlap
    chunk_size, overlap = pick_chunk_params(text)

    # 6) Разбиваем на чанки
    chunks = split_chunks_by_tokens(text, chunk_size, overlap)

    # 7) Возвращаем JSON-ответ (ensure_ascii=False для корректного отображения кириллицы)
    response_body = json.dumps({"chunks": chunks}, ensure_ascii=False)
    return Response(response_body, content_type="application/json")


if __name__ == "__main__":
    # Запуск локального сервера для отладки
    app.run(host="0.0.0.0", port=5555, debug=True)
