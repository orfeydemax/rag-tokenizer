from flask import Flask, request, Response
import tiktoken
import json

app = Flask(__name__)

def clean_text(text):
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\f', ' ')
    while '  ' in text:
        text = text.replace('  ', ' ')
    return text.strip()

def pick_chunk_params(text, encoding_name="cl100k_base"):
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

def split_chunks_by_tokens(text, chunk_size, overlap):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        try:
            decoded = encoding.decode(chunk)
        except Exception:
            decoded = ""
        chunks.append(decoded)
    return chunks

@app.route('/split', methods=['POST'])
def split():
    data = request.get_json()
    text = data.get("text", "")
    text = clean_text(text)
    chunk_size, overlap = pick_chunk_params(text)
    chunks = split_chunks_by_tokens(text, chunk_size, overlap)
    # Возвращаем JSON с поддержкой любого языка (ensure_ascii=False)
    response = json.dumps({"chunks": chunks}, ensure_ascii=False)
    return Response(response, content_type="application/json")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
