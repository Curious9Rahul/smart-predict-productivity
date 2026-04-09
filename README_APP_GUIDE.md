# Smart App Predictor 🎯

An AI-powered behavioral intelligence system that predicts your next app and suggests productive alternatives. Uses real Android OS usage stats and ML models to understand your app patterns.

## Features 🌟

- 📊 **Live App Usage Analytics** - Real-time data from Android UsageStatsManager
- 🤖 **Intelligent Predictions** - ML models predict your next app with context awareness
- ⚠️ **Distraction Detection** - AI identifies and alerts you to distraction patterns
- 💡 **Smart Interventions** - Personalized app recommendations based on time and behavior
- 👤 **Behavioral Profiling** - Clusters users into behavior types (Night Binger, Focus Pro, etc.)
- 📱 **Beautiful UI** - Dark theme, real-time charts, smooth navigation

## Architecture 🏗️

### Backend (Python/Flask)
- Trained ML models (Random Forest, Gradient Boosting, K-Means)
- Markov chain transition prediction
- REST API endpoints for predictions, analysis, and profiling
- User profile management

### Frontend (Flutter)
- Cross-platform mobile app (Android/iOS)
- Native Android integration via MethodChannel
- Real-time app usage data collection
- Configurable backend IP for local network deployment

### Data Flow
```
Phone (Flutter App)
  ↓ (UsageStatsManager API)
  ├→ Collects real app usage data
  ├→ Sends to Backend (local network)
  ↓
Backend (Flask)
  ├→ /predict - Next app prediction
  ├→ /analyze - Behavioral analysis  
  ├→ /distraction - Distraction detection
  ├→ /productivity - User profiles
  ↓
Phone (Flutter App)
  └→ Displays predictions & recommendations
```

---

## Quick Start (5 easy steps)

### Step 1: Start Backend Server

**Windows:**
```bash
cd d:\combo\smart_app_predictor
double-click start_backend.bat
```

**Mac/Linux:**
```bash
cd combo/smart_app_predictor
python app.py
```

Server starts at `http://localhost:5000`

### Step 2: Find Your Computer's IP

**Windows:**
```bash
ipconfig
```
Look for: **IPv4 Address** (e.g., `192.168.1.100`)

**Mac/Linux:**
```bash
ifconfig
```

### Step 3: Build Flutter APK

**Windows:**
```bash
cd d:\combo\smart_app_predictor\flutter_frontend
double-click build_apk.bat
```

**Mac/Linux:**
```bash
cd combo/smart_app_predictor/flutter_frontend
flutter build apk --release
```

APK location: `build/app/outputs/flutter-apk/app-release.apk`

### Step 4: Install on Phone

**Option A - USB Cable (Recommended):**
1. Connect phone via USB
2. Enable USB Debugging (Settings → Developer Options → USB Debugging)
3. Run: `flutter install`

**Option B - Manual:**
1. Copy APK to phone (email, cloud storage, USB)
2. Open file → Install
3. Allow installation from unknown sources

### Step 5: Configure App

1. Open **Smart App Predictor** app
2. Go to **Settings** tab
3. Enter: `http://192.168.X.X:5000` (your computer IP)
4. Tap **Save Configuration**
5. Grant **Usage Access** permission when prompted
6. Go to **Dashboard** → Tap **Sync** to start

---

## Usage Guide 📖

### Dashboard
- **Sync Button** - Fetch live app usage from phone
- **Real-time Analysis** - See your current productivity score
- **Distraction Alerts** - Get warned about distraction patterns
- **Recent Apps** - View your recent app sessions

### Analytics  
- **Pie Chart** - App usage breakdown by category (Social, Productivity, Entertainment)
- **Bar Chart** - Daily usage by individual app
- **Category Stats** - Time spent in each category

### Predict
- **Select Context** - Choose previous app, time of day, day of week, usage frequency
- **AI Prediction** - ML predicts next app you'll open
- **Confidence Score** - Percentage certainty of prediction
- **Smart Intervention** - Recommended productive app instead

### Profile
- **Behavioral Type** - Which user profile cluster you belong to
- **Peak Hours** - When you're most active
- **Productivity Score** - Overall productivity metric (0-100)
- **Distraction Rate** - % of sessions that are distraction
- **Top Apps** - Your most-used apps

### Settings
- **Backend IP Config** - Update server connection details
- **Connection Testing** - Verify Flask backend is accessible
- **Setup Guide** - Step-by-step instructions

---

## ML Models Explained 🧠

### Random Forest (Prediction)
- Trained on 11 contextual features
- Predicts next app from previous app + time + day + frequency
- Handles multiple decision paths for better accuracy

### Gradient Boosting (Distraction Detection)
- 12-feature model detecting distraction patterns
- Identifies rapid app switching sequences
- Real-time alerts on Dashboard

### K-Means Clustering (User Profiling)
- 4 behavioral clusters:
  - 🌙 **Night Binger** - Heavy evening/night usage, entertainment-focused
  - 💼 **Focus Pro** - Strong morning productivity, work-oriented
  - 📱 **Social Butterfly** - Consistent social media engagement
  - ⚖️ **Balanced User** - Mixed usage patterns

### Markov Chain (Transition Matrix)
- Models app switching probabilities
- Captures sequential patterns (Instagram → YouTube is common)
- Hyper-smoothed for better generalization

---

## Project Structure 📁

```
smart_app_predictor/
├── app.py                          # Flask backend server
├── train.py                        # ML model training script
├── utils.py                        # Helper functions
├── requirements.txt                # Python dependencies
├── start_backend.bat               # Windows backend launcher
│
├── model/                          # Trained ML models
│   ├── rf_model.pkl                # Random Forest predictor
│   ├── markov_matrix.pkl           # Markov transition matrix
│   ├── kmeans_model.pkl            # K-Means clusterer
│   ├── distraction_model.pkl       # Distraction detector
│   ├── config.json                 # Encoding mappings
│   └── user_profiles.json          # User profile data
│
├── dataset/
│   └── app_usage.csv               # Training data
│
├── templates/
│   └── index.html                  # Web dashboard
│
└── flutter_frontend/               # Mobile app
    ├── BUILD_AND_INSTALL_GUIDE.md  # Detailed setup guide
    ├── build_apk.bat               # Windows APK builder
    ├── pubspec.yaml                # Flutter dependencies
    │
    ├── lib/
    │   ├── main.dart               # App entry point
    │   ├── config/
    │   │   └── api_config.dart     # Configurable backend IP
    │   ├── services/
    │   │   └── app_usage_service.dart  # Native Android bridge
    │   └── screens/
    │       ├── dashboard_screen.dart
    │       ├── analytics_screen.dart
    │       ├── prediction_screen.dart
    │       ├── intervention_screen.dart
    │       └── settings_screen.dart
    │
    ├── android/
    │   ├── app/
    │   │   ├── build.gradle.kts
    │   │   └── src/main/
    │   │       ├── kotlin/
    │   │       │   └── MainActivity.kt  # UsageStatsManager integration
    │   │       └── AndroidManifest.xml
    │   └── gradle.properties
    │
    └── build/                      # Build output (after flutter build apk)
        └── app/outputs/flutter-apk/
            └── app-release.apk     # Your installable APK!
```

---

## Troubleshooting 🔧

### Backend Issues

**"Flask not found"**
```bash
pip install flask flask-cors
```

**"Port 5000 already in use"**
```bash
python app.py --port 5001
```

**"Cannot import models"**
- Ensure trained model files exist in `model/` directory
- Run `python train.py` to retrain models

### App Issues

**"Cannot connect to server"**
- Check backend is running: `python app.py`
- Update IP in Settings (must be computer's local IP, not localhost)
- Phone and computer must be on same WiFi

**"No app usage data"**
- Grant "Usage Access" permission (Settings → Apps → Smart App Predictor)
- Android 6.0+ requires explicit permission grant
- Wait 2-3 minutes after installation

**"APK won't install"**
- Enable "Unknown Sources" (Settings → Security)
- Check Android version (minimum 5.0)
- Try a different phone or emulator

**"App crashes on launch"**
```bash
flutter clean
flutter pub get
flutter build apk --release
```

### Connection Testing

Test backend endpoint manually:
```bash
curl http://192.168.X.X:5000/predict -X POST -H "Content-Type: application/json" -d "{\"previous_app\": \"Instagram\", \"time_of_day\": \"Night\", \"day_of_week\": \"Monday\", \"usage_frequency\": \"Medium\"}"
```

---

## API Reference 🌐

All endpoints expect JSON requests and return JSON responses.

### `POST /predict`
Predict next app based on context.

**Request:**
```json
{
  "previous_app": "Instagram",
  "time_of_day": "Night",
  "day_of_week": "Monday",
  "usage_frequency": "Medium"
}
```

**Response:**
```json
{
  "predicted_app": "YouTube",
  "confidence": 0.78,
  "category": "Entertainment",
  "intervention": "🌙 Enable bedtime mode",
  "warning": "Late scrolling disrupts sleep"
}
```

### `POST /analyze`
Analyze app sequence for distraction and productivity.

**Request:**
```json
{
  "sessions": [
    {"app": "Instagram", "duration": 25, "time": "Night"},
    {"app": "YouTube", "duration": 40, "time": "Night"}
  ]
}
```

**Response:**
```json
{
  "productivity_score": 28.5,
  "behavior_type": "Night Binger",
  "distraction_level": "High",
  "score_trend": "Declining"
}
```

### `POST /distraction`
Detect distraction patterns in app sequence.

**Request:**
```json
{
  "sequence": ["Instagram", "YouTube", "Gmail", "Instagram"]
}
```

**Response:**
```json
{
  "distraction_detected": true,
  "alert": "⚠️ Rapid app switching detected"
}
```

### `GET /productivity?user_id=user_001`
Get user behavioral profile.

**Response:**
```json
{
  "persona": "Night Binger",
  "peak_time": "Night",
  "avg_duration": 42.5,
  "avg_productivity_score": 30.0,
  "distraction_rate_pct": 68.0,
  "top_apps": {
    "Netflix": 182,
    "YouTube": 145,
    "Instagram": 120
  }
}
```

---

## Requirements 📋

### Backend
- Python 3.8+
- Flask, scikit-learn, pandas, numpy

### Frontend  
- Flutter 3.0+
- Android SDK (API 21+)
- Java 11+ (JDK)

### Device
- Android 5.0+ phone
- WiFi connection to computer

---

## License 📜

Educational project. Feel free to modify and use for your own purposes.

---

## Next Steps 🚀

1. ✅ Run `start_backend.bat` or `python app.py`
2. ✅ Find your computer's IP with `ipconfig`
3. ✅ Run `build_apk.bat` or `flutter build apk --release`
4. ✅ Install APK on phone
5. ✅ Configure Settings with backend IP
6. ✅ Start exploring your app usage patterns!

---

## Questions?

- Check `BUILD_AND_INSTALL_GUIDE.md` for detailed setup steps
- Run `flutter doctor` to diagnose Flutter issues
- Check that backend server is running and accessible
- Verify phone has Usage Stats permission granted

Enjoy! 🎉
