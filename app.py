import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from rag import ingest, ask

load_dotenv()

app = Flask(__name__)

UPLOAD_DIR = "./uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"Unsupported file type: .{ext}"}), 400

    path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(path)

    try:
        num_chunks = ingest(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(path)

    return jsonify({"status": "ok", "filename": file.filename, "chunks": num_chunks})


@app.route("/ask", methods=["POST"])
def query():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        answer = ask(question)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
