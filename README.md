### 🎭 Video Face Swap (InsightFace + Gradio)

The goal of this WebUI is to make it as easy as possible to create deepfakes using the InsightFace framework.
This tool provides a simple interface to swap faces in videos with just a few clicks.

### ✨ Features

🔹 Swap a single face in a video.
🔹 Click-to-select which face should be swapped.
🔹 Adjust face detection tolerance so only the chosen face is modified.
🔹 Runs locally or on Google Colab with GPU support.

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/dhrutitagline/Video-Face-Swap.git
cd Video-Face-Swap
```

### 2️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Model Weights

#### 🔹 Local PC
```bash
curl -L -o inswapper_128.onnx https://huggingface.co/xingren23/comfyflow-models/resolve/main/insightface/inswapper_128.onnx
```

#### 🔹 Google Colab (GPU)
```bash
!wget https://huggingface.co/xingren23/comfyflow-models/resolve/main/insightface/inswapper_128.onnx
```
Place the downloaded file inside the project directory.

### 🖼️ Usage
Run the application with:
```bash
python main.py
```

Once started, Gradio will launch a local web UI:
👉 http://127.0.0.1:7860
