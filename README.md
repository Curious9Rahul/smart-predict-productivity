# Smart App Usage Predictor

A full-stack ML project that predicts the next mobile app a user will open using **Random Forest** and **Markov Chain** models.

---

## 🏗 Project Structure

```
smart_app_predictor/
├── dataset/            ← Auto-generated synthetic CSV
├── model/              ← Trained model pickle files
├── utils.py            ← Dataset generator
├── train.py            ← ML training (RF + Markov)
├── app.py              ← Flask REST API
├── requirements.txt
└── flutter_frontend/   ← Flutter mobile app
    ├── lib/
    │   ├── main.dart
    │   ├── models/
    │   ├── screens/    ← Home, Stats, History
    │   ├── services/   ← API service + Provider
    │   └── widgets/    ← Shared UI components
    └── pubspec.yaml
```

---

## ⚙️ Setup — Backend (Python + Flask)

### 1. Install dependencies
```bash
cd smart_app_predictor
pip install -r requirements.txt
```

### 2. Generate dataset & train models
```bash
python utils.py    # creates dataset/app_usage.csv
python train.py    # trains models, saves to model/
```

### 3. Start Flask API
```bash
python app.py
```
API runs at `http://0.0.0.0:5000`

---

## 📱 Setup — Flutter App

### Prerequisites
- Flutter SDK ≥ 3.0
- Android Studio / VS Code with Flutter plugin

### 1. Install packages
```bash
cd flutter_frontend
flutter pub get
```

### 2. Configure backend URL in `lib/services/api_service.dart`
| Scenario | URL |
|---|---|
| Android Emulator | `http://10.0.2.2:5000` (default) |
| Physical Android Device | `http://<YOUR_PC_LAN_IP>:5000` |
| Web/Desktop | `http://localhost:5000` |

### 3. Run the app
```bash
flutter run
```

---

## 🌐 API Reference

### `POST /predict`
```json
{
  "previous_app": "Instagram",
  "time_of_day": "Night",
  "day_of_week": "Monday",
  "usage_frequency": "High",
  "model": "random_forest"   // or "markov"
}
```
**Response:**
```json
{
  "predicted_app": "YouTube",
  "confidence": 0.87,
  "top_3": [
    {"app": "YouTube", "probability": 0.87},
    {"app": "Instagram", "probability": 0.08},
    {"app": "Netflix", "probability": 0.05}
  ],
  "model_used": "random_forest",
  "alternative_suggestion": "LinkedIn Learning"
}
```

### `GET /apps`
Returns available options for dropdowns.

### `GET /health`
Health check.

---

## 🎤 Viva Talking Points

1. *"We converted sequential app usage into a supervised learning problem by shifting app sequences to create the target variable."*
2. *"We compared ensemble learning (Random Forest) with probabilistic sequence modelling (Markov Chain) for next-app prediction."*
3. *"Future scope: replace synthetic data with real-time Android UsageStats API data."*
