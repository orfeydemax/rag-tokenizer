from flask import Flask, request, Response
import tiktoken
import json

app = Flask(__name__)

def clean_text(text: str) -> str:
    """
    Очищает входной текст от артефактов:
    - заменяет "null" (в любой регистровой комбинации) на пробел
    - убирает символы перевода строки, табуляции и формы FF
    - сводит подряд идущие пробелы к одному
    """
    # 1) Убираем BOM-метку в начале, если она есть
    if text.startswith("\ufeff"):
        text = text[1:]

    # 2) Заменяем все "null" (в разных регистрах) на пробел
    #    Регистрируем все вхождения "null" (регистр игнорируется).
    text = text.replace("null", " ").replace("NULL", " ").replace("Null", " ")

    # 3) Убираем переводы строк, табуляции, form feed
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\f', ' ')

    # 4) Сводим любые подряд идущие пробелы к одиночному
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
    Разбивает текст на чанки по chunk_size токенов с overlap (перекрытием) токенов.
    Возвращает список decoded-строк (раскодированных токенов).
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks: list[str] = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size  # на случай, если overlap >= chunk_size

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i:i + chunk_size]
        try:
            decoded = encoding.decode(chunk_tokens)
        except Exception:
            # если вдруг декодирование упадёт, возвращаем пустую строку
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
    try:
        data = request.get_json(force=True)
    except Exception as e:
        # если вход невалидный JSON
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

    # Забираем текст
    raw_text = data.get("text")
    if raw_text is None:
        return Response(
            json.dumps({"error": "В теле запроса отсутствует ключ 'text'"}, ensure_ascii=False),
            status=400,
            content_type="application/json"
        )

    # 1) Очищаем текст от мелких артефактов
    text = clean_text(str(raw_text))

    # 2) Подбираем параметры chunk_size и overlap
    chunk_size, overlap = pick_chunk_params(text)

    # 3) Разбиваем на чанки
    chunks = split_chunks_by_tokens(text, chunk_size, overlap)

    # 4) Возвращаем результат в виде JSON, при этом ensure_ascii=False
    response_body = json.dumps({"chunks": chunks}, ensure_ascii=False)
    return Response(response_body, content_type="application/json")

if __name__ == "__main__":
    # Запуск локального сервера для отладки
    app.run(host="0.0.0.0", port=5555, debug=True)
