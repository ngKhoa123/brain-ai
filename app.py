from flask import Flask, request, render_template, send_from_directory, url_for
import os
import torch
import joblib

from preprocess import extract_hog_single, preprocess_dl_for_model

from torchvision import models, transforms
import torch.nn as nn
from werkzeug.utils import secure_filename

import gdown

# ===== DOWNLOAD DL MODEL =====
if not os.path.exists("model_dl.pth"):
    gdown.download(
        "https://drive.google.com/uc?id=1IdjMDP6dvNF3WBx2RaOwqTilOo_DMy3B",
        "model_dl.pth",
        quiet=False
    )

# ===== DOWNLOAD SVM MODEL =====
if not os.path.exists("model_svm.pkl"):
    gdown.download(
        "https://drive.google.com/uc?id=1vW4qe4yTSrqypsdPm4fwsmTSB_TlR76i",
        "model_svm.pkl",
        quiet=False
    )
# =========================
# INIT APP
# =========================
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# SERVE IMAGE
# =========================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# =========================
# LOAD MODEL (⚠️ LOAD 1 LẦN)
# =========================
model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(1280, 4)

model.load_state_dict(torch.load("model_dl.pth", map_location="cpu"))
model.eval()

svm_pipeline = joblib.load("model_svm.pkl")

class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

# =========================
# ROUTE
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = secure_filename(file.filename)
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            # ===== DL =====
            dl_results = []
            img = preprocess_dl_for_model(path, test_transform)

            if img is not None:
                with torch.no_grad():
                    output = model(img)
                    probs = torch.softmax(output, dim=1)[0]

                for i, p in enumerate(probs):
                    dl_results.append({
                        "label": class_names[i],
                        "prob": round(p.item() * 100, 2)
                    })

                dl_results = sorted(dl_results, key=lambda x: x["prob"], reverse=True)
                dl_label = dl_results[0]["label"]
                dl_conf = dl_results[0]["prob"]
            else:
                dl_label, dl_conf = "Error", 0

            # ===== SVM =====
            svm_results = []
            try:
                feat = extract_hog_single(path)
                probs = svm_pipeline.predict_proba(feat)[0]

                for i, p in enumerate(probs):
                    svm_results.append({
                        "label": class_names[i],
                        "prob": round(p * 100, 2)
                    })

                svm_results = sorted(svm_results, key=lambda x: x["prob"], reverse=True)
                svm_label = svm_results[0]["label"]
                svm_conf = svm_results[0]["prob"]

            except Exception as e:
                print("SVM ERROR:", e)
                svm_label, svm_conf = "Error", 0

            return render_template(
                "index.html",
                dl_prediction=dl_label,
                dl_confidence=dl_conf,
                dl_results=dl_results,
                svm_prediction=svm_label,
                svm_confidence=svm_conf,
                svm_results=svm_results,
                image_url=url_for('uploaded_file', filename=filename)
            )

    return render_template("index.html")


# =========================
# RUN (QUAN TRỌNG KHI DEPLOY)
# =========================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  
    app.run(host="0.0.0.0", port=port)