# Krushi Rakshak — Backend + Website Prototype

SIH 2026 · Problem Statement **SIH26131** — "Early detection and management of
crop diseases and pest infestations" · Team **SIH2026-0151**

This is a working, end-to-end prototype of the system described in the
team's PPT: a farmer uploads a crop/leaf photo, the system detects a
likely disease or pest, scores the risk, gives simple advice, and logs
the report so agriculture officers can spot hotspots.

## 1. Project structure

```
krushi-rakshak/
├── app.py              # Flask backend — all API routes + SQLite logic
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # The website (upload form + dashboard)
├── static/
│   ├── style.css        # Visual design
│   └── script.js         # Talks to the backend API
├── uploads/              # Uploaded photos are saved here
└── krushi_rakshak.db     # SQLite database (created automatically on first run)
```

## 2. How to run it

```bash
cd krushi-rakshak
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in a browser. Upload any leaf/crop
photo (or any image, for testing), and the site will show a detected
disease, a risk level, and advice — and log it to the history and
hotspot tables below the form.

## 3. How this maps to the PPT's System Architecture slide

| PPT box                     | This project                                      |
|------------------------------|---------------------------------------------------|
| Mobile App (Frontend)        | `templates/index.html` + `static/` (upload form, results, dashboard) |
| Backend Server & API         | `app.py` — Flask routes `/api/detect`, `/api/history`, `/api/hotspots`, `/api/feedback` |
| AI/ML Model (Detection)      | `run_ai_detection()` inside `app.py`              |
| Database (History & Logs)    | SQLite, table `reports` in `krushi_rakshak.db`    |
| Feedback & Model Improvement | `/api/feedback` route + `feedback_log` table       |

The PPT's flow **Capture → Detect (AI) → Advise → Act Early** is exactly
what happens inside the `/api/detect` route, step by step, with comments
marking each stage.

## 4. About the AI part (important for judges/mentors)

`run_ai_detection()` currently uses a simple, transparent color-ratio
heuristic on the photo (via Pillow) instead of a trained neural network.
This was a deliberate choice for the prototype: it needs **no dataset,
no GPU, and no training time**, so the *entire pipeline* — upload,
detection, risk scoring, advice, database logging, dashboard — runs
today and can be demoed live.

The function is written so a real model drops in without touching
anything else:

```python
from tensorflow.keras.models import load_model
model = load_model("plant_disease_mobilenet.h5")

def run_ai_detection(image_path):
    img = preprocess(image_path)
    prediction = model.predict(img)
    class_id = prediction.argmax()
    confidence = float(prediction.max())
    disease_info = DISEASE_LIBRARY[class_id]
    return disease_info, confidence
```

Everything downstream (database, risk level, dashboard, hotspots) reads
only the `(disease_info, confidence)` this function returns, so swapping
in MobileNet/EfficientNet/YOLO trained on the PlantVillage dataset (as
named in the PPT's References slide) is a one-function change.

## 5. Extending toward the full PPT vision

- **Weather + GIS inputs**: add fields to the `/api/detect` form and a
  `weather` column to `reports`; factor them into `confidence_to_risk()`.
- **Multilingual support**: translate `templates/index.html` strings;
  keep the API and database exactly as they are.
- **C++ backend**: the API contract (routes, request/response JSON
  shapes) in this file is the spec — a C++ HTTP server can implement the
  same four routes and the frontend needs no changes.
- **District/state hotspot map**: `/api/hotspots` already returns
  location + disease + count; plug that into a map library (e.g.
  Leaflet) on the frontend.
