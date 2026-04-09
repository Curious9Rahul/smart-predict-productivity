# Quick Reference Card 🎯

## Before You Start
- [ ] Flask server running (`start_backend.bat` or `python app.py`)
- [ ] Found your computer's IP (`ipconfig` → IPv4 Address)
- [ ] Flutter installed on your machine
- [ ] Android SDK/Java 11+ installed
- [ ] USB cable or emulator ready

---

## 3-Minute Build Process

### Windows:
```
1. cd d:\combo\smart_app_predictor\flutter_frontend
2. double-click build_apk.bat
3. Wait 5-10 minutes...
4. APK ready at: build\app\outputs\flutter-apk\app-release.apk
```

### Mac/Linux:
```
1. cd combo/smart_app_predictor/flutter_frontend
2. flutter build apk --release
3. APK ready at: build/app/outputs/flutter-apk/app-release.apk
```

---

## Installation Options (Pick One)

### Option 1: USB Cable (Easiest)
```
1. Connect phone via USB
2. Enable USB Debug (Settings → Developer Options)
3. flutter install
```

### Option 2: Manual Transfer
```
1. Email/cloud/USB the APK to your phone
2. Open file → Install
3. Allow unknown sources
```

### Option 3: adb
```
adb install build/app/outputs/flutter-apk/app-release.apk
```

---

## Post-Installation (Do This First!)

1. **Open Settings tab** in app
2. **Enter backend IP**: `http://192.168.X.X:5000`
3. **Tap Save Configuration**
4. **Grant Usage Access permission** when prompted
5. **Go to Dashboard** → Tap **Sync**

---

## Endpoints (For Testing)

```
Backend running? → http://192.168.X.X:5000 (in browser)
Predict API → POST http://192.168.X.X:5000/predict
Analyze API → POST http://192.168.X.X:5000/analyze  
Distraction API → POST http://192.168.X.X:5000/distraction
Profile API → GET http://192.168.X.X:5000/productivity?user_id=user_001
```

---

## Common Issues

| Problem | Solution |
|---------|----------|
| "Flutter not found" | Install Flutter SDK, add to PATH |
| "Cannot connect" | Update IP in Settings, check backend running |
| "No usage data" | Grant Usage Access permission, force sync |
| "APK won't install" | Enable Unknown Sources, check Android version |
| "Port 5000 in use" | `python app.py --port 5001` |

---

## File Locations

- **Backend**: `d:\combo\smart_app_predictor\app.py`
- **APK Output**: `flutter_frontend\build\app\outputs\flutter-apk\app-release.apk`
- **Config**: App stores IP in phone's SharedPreferences (automatic)
- **Logs**: Check phone's Logcat if issues: `adb logcat`

---

## App Settings (After Install)

| Tab | What It Does |
|-----|-------------|
| 📊 Dashboard | Real-time sync, productivity score, last app used |
| 📈 Analytics | Pie/bar charts of usage by app and category |
| 🧠 Predict | Select context, see next app prediction + intervention |
| 👤 Profile | Behavioral type, peak hours, top apps, distraction rate |
| ⚙️ Settings | **Configure backend IP** ← START HERE |

---

## What Real-Time Data You Get

✅ **Collected from your phone:**
- Current app name
- Time in foreground (minutes)
- Last time used
- Package name

✅ **Sent to backend:**
- Aggregated session data
- Context (time of day, day, frequency)

✅ **NOT collected:**
- App content/screenshots
- Personal files
- Passwords
- Location data

---

## Support Checklist

1. Backend running? `python app.py` → shows `Running on http://0.0.0.0:5000` ✓
2. Right IP? `ipconfig` → matches Settings IP ✓
3. Permission granted? Settings → Apps → Smart App Predictor → Permissions ✓
4. APK correct version? Check timestamp of app-release.apk ✓
5. Phone connected? USB debug on, device shows in `adb devices` ✓

---

## Let's Go! 🚀

1. Start backend
2. Build APK  
3. Install on phone
4. Enter IP in Settings
5. Tap Sync on Dashboard
6. Watch the magic happen ✨

Questions? Check `BUILD_AND_INSTALL_GUIDE.md` for detailed steps!
