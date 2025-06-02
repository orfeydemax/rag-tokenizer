from flask import Flask, request, Response
import tiktoken
import json
import re
import unicodedata

app = Flask(__name__)

def clean_text(text: str) -> str:
    """
    Очищает входной текст от артефактов:
    1) Убирает BOM-метку в начале (U+FEFF).
    2) Заменяет символ Replacement Character '�' (U+FFFD) на пробел.
    3) Заменяет 'null' (в любом регистре) на пробел.
    4) Убирает управляющие символы (U+0000–U+001F, U+007F–U+009F), табуляции, переводы строк и form-feed.
    5) Сводит подряд идущие пробелы к одному.
    """
    # 1) Убираем BOM-метку в начале, если есть
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2) Заменяем символ '�' (Unicode Replacement Character) на пробел
    #    Иногда встречаются и другие похожие артефакты, но U+FFFD—самый частый.
    text = text.replace("\ufffd", " ")

    # 3) Заменяем 'null' (в любом регистре) на пробел
    #    Чтобы удалить "null", "NULL", "Null" и т.д.
    text = re.sub(r'null', ' ', text, flags=re.IGNORECASE)

    # 4) Убираем управляющие и непечатаемые символы:
    #    - U+0000–U+001F (C0 control codes)
    #    - U+007F–U+009F (DEL и C1 control codes)
    #    Также удаляем табуляцию, переводы строк и form-feed (хотя они попадают в U+0000–U+001F).
    #    Здесь мы просто заменяем их на пробел, чтобы не сломать границы слов.
    text = re.sub(r'[\u0000-\u001F\u007F-\u009F]', ' ', text)

    # 5) Сводим подряд идущие пробелы к одному
    #    Делать это лучше циклом, чтобы убрать любые двойные и более пробелы.
    while '  ' in text:
        text = text.replace('  ', ' ')

    return text.strip()

def pick_chunk_params(text: str, encoding_name: str = "cl100k_base") -> tuple[int, int]:
    """
    Автоподбор параметров chunk_size и overlap исходя из длины текста в токенах.
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
    Разбивает текст на чанки по chunk_size токенов с overlap (перекрытием).
    Возвращает список декодированных строк (каждый чанк — это кусок текста).
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks: list[str] = []

    step = chunk_size - overlap
    if step <= 0:
        # Если overlap >= chunk_size, чтобы не было бесконечного цикла
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
    API-метод POST /split
    Принимает JSON: { "text": "<длинный текст>" }
    Возвращает JSON: { "chunks": [ "<чанк 1>", "<чанк 2>", … ] }
    """
    # 1) Парсим JSON
    try:
        data = request.get_json(force=True)
    except Exception as e:
        return Response(
            json.dumps({"error": "Невозможно распарсить JSON", "exception": str(e)}, ensure_ascii=False),
            status=400,
            content_type="application/json"
        )

    if not isinstance(data, dict):
        return Response(
            json.dumps({"error": "Ожидался JSON-объект"}, ensure_ascii=False),
            status=400,
            content_type="application/json"
        )

    # 2) Берем поле "text"
    raw_text = data.get("text")
    if raw_text is None:
        return Response(
            json.dumps({"error": "В теле запроса отсутствует ключ 'text'"}, ensure_ascii=False),
            status=400,
            content_type="application/json"
        )

    # 3) Очищаем текст от артефактов
    text = clean_text(str(raw_text))

    # 4) Подбираем параметры chunk_size и overlap
    chunk_size, overlap = pick_chunk_params(text)

    # 5) Разбиваем текст на чанки
    chunks = split_chunks_by_tokens(text, chunk_size, overlap)

    # 6) Возвращаем JSON с чанками (ensure_ascii=False, чтобы кириллица отображалась нормально)
    response_body = json.dumps({"chunks": chunks}, ensure_ascii=False)
    return Response(response_body, content_type="application/json")

if __name__ == "__main__":
    # Запускаем локальный Flask-сервер для отладки
    app.run(host="0.0.0.0", port=5555, debug=True)
