from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS
from src.predict import predict_image
import os

# =========================
# KONFIGURASI PATH
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "temp")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# =========================
# INIT FLASK
# =========================
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

CORS(app)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# HELPER FUNCTION
# =========================
def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# ROUTE HALAMAN WEB
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/deteksi")
def deteksi():
    return render_template("deteksi.html")

@app.route("/edukasi")
def edukasi():
    return render_template("edukasi.html")

@app.route("/tentang")
def tentang():
    return render_template("tentang.html")

# =========================
# API PREDIKSI
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "File gambar tidak ditemukan"}), 400

    image_file = request.files["image"]

    if image_file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    if not allowed_file(image_file.filename):
        return jsonify({"error": "Format file tidak didukung"}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        # Simpan file sementara
        image_file.save(image_path)

        # Prediksi CNN
        label, confidence = predict_image(image_path)

        return jsonify({
            "prediction": label,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        print("❌ ERROR:", e)
        return jsonify({"error": "Gagal melakukan prediksi"}), 500

    finally:
        # Hapus file sementara
        if os.path.exists(image_path):
            os.remove(image_path)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
