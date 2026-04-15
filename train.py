"""
train.py — Unified Intelligence Loop v2
Single Random Forest model that predicts:
  1. next_app           (classification)
  2. distraction_level  (classification: Low / Moderate / High)
  3. persona            (classification)
  4. productivity_score (regression → bucketed to int)

Run: python train.py
Saves: model/unified_model.joblib  +  model/config.json
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

from utils import generate_synthetic_data

# ─── Mappings ─────────────────────────────────────────────────────────────────
APP_MAP  = {'Instagram': 0, 'WhatsApp': 1, 'YouTube': 2,
            'Chrome': 3, 'Gmail': 4, 'LinkedIn': 5, 'Netflix': 6}
TIME_MAP = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
DAY_MAP  = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6}
CAT_MAP  = {'Social': 0, 'Productivity': 1, 'Entertainment': 2}
FREQ_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
PERSONA_MAP = {
    'Night Binger': 0, 'Morning Worker': 1, 'Social Butterfly': 2,
    'Balanced User': 3, 'Notification Addict': 4, 'Weekend Warrior': 5
}
DISTRACTION_MAP = {'None': 0, 'Moderate': 1, 'High': 2}

FEATURE_NAMES = [
    'prev_enc', 'time_enc', 'day_enc', 'freq_enc', 'cat_enc',
    'usage_duration', 'battery_level', 'screen_on_time_today',
    'session_number', 'is_weekend', 'switch_count_last_hour',
    'notification_triggered'
]

APP_CAT = {
    'Instagram': 'Social',   'WhatsApp': 'Social',    'LinkedIn': 'Social',
    'Gmail': 'Productivity', 'Chrome': 'Productivity',
    'YouTube': 'Entertainment', 'Netflix': 'Entertainment',
}


def _distraction_label(row) -> str:
    """Derive 3-class distraction label from dataset flags."""
    if not row['distraction_detected']:
        return 'None'
    if row['distraction_pattern'] in ('Social Loop', 'App Hopping'):
        return 'High'
    return 'Moderate'


def train():
    os.makedirs('model', exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────────
    if not os.path.exists('dataset/app_usage.csv'):
        print("[*] Generating synthetic dataset (15,000 rows)...")
        generate_synthetic_data(15000)
    df = pd.read_csv('dataset/app_usage.csv')
    print(f"[OK] Loaded {len(df):,} rows x {len(df.columns)} columns")

    # ── 2. Feature engineering ─────────────────────────────────────────────
    df['prev_enc']  = df['previous_app'].map(APP_MAP)
    df['time_enc']  = df['time_of_day'].map(TIME_MAP)
    df['day_enc']   = df['day_of_week'].map(DAY_MAP)
    df['freq_enc']  = df.get('usage_frequency', pd.Series('Medium', index=df.index)).map(FREQ_MAP).fillna(1).astype(int)
    df['cat_enc']   = df['app_category'].map(CAT_MAP)

    # ── 3. Targets ─────────────────────────────────────────────────────────
    df['next_app_enc']     = df['next_app'].map(APP_MAP)
    df['distract_enc']     = df.apply(_distraction_label, axis=1).map(DISTRACTION_MAP)
    df['persona_enc']      = df['persona'].map(PERSONA_MAP)
    # Productivity score → bucketed 0/1/2 (Low / Medium / High)
    df['prod_bucket']      = pd.cut(df['productivity_score'],
                                    bins=[0, 40, 65, 100],
                                    labels=[0, 1, 2]).astype(int)

    df.dropna(subset=FEATURE_NAMES + ['next_app_enc', 'distract_enc', 'persona_enc', 'prod_bucket'], inplace=True)

    X = df[FEATURE_NAMES].values
    Y = df[['next_app_enc', 'distract_enc', 'persona_enc', 'prod_bucket']].values

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.15, random_state=42, stratify=Y[:, 0]
    )

    # ── 4. Unified Random Forest ──────────────────────────────────────────
    print("\n[*] Training Unified Random Forest (multi-output)...")
    base_rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    from sklearn.multioutput import MultiOutputClassifier
    model = MultiOutputClassifier(base_rf, n_jobs=-1)
    model.fit(X_train, Y_train)

    # ── 5. Accuracy per target ─────────────────────────────────────────────
    Y_pred = model.predict(X_test)
    targets = ['next_app', 'distraction', 'persona', 'prod_bucket']
    print("\n[ACCURACY] Per-target accuracy:")
    for i, t in enumerate(targets):
        acc = accuracy_score(Y_test[:, i], Y_pred[:, i])
        print(f"   {t:<18}: {acc:.2%}")

    next_app_acc = accuracy_score(Y_test[:, 0], Y_pred[:, 0])
    print(f"\n[OK] Primary (next_app) accuracy: {next_app_acc:.2%}")

    # ── 6. Feature importances (from next_app estimator) ──────────────────
    importances = model.estimators_[0].feature_importances_
    feat_imp = dict(zip(FEATURE_NAMES, [round(float(v), 4) for v in importances]))
    print("\n[INFO] Feature importances (next_app):")
    for k, v in sorted(feat_imp.items(), key=lambda x: -x[1]):
        print(f"   {k:<30}: {v:.4f}")

    # ── 7. Save model + config ─────────────────────────────────────────────
    joblib.dump(model, 'model/unified_model.joblib', compress=3)
    print("[SAVED] model/unified_model.joblib")

    config = {
        'version': '2.0.0-UnifiedRF',
        'features': FEATURE_NAMES,
        'app_map':  APP_MAP,
        'time_map': TIME_MAP,
        'day_map':  DAY_MAP,
        'cat_map':  CAT_MAP,
        'freq_map': FREQ_MAP,
        'persona_map': PERSONA_MAP,
        'distraction_map': DISTRACTION_MAP,
        'targets': targets,
        'feature_importances': feat_imp,
    }
    with open('model/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("[SAVED] model/config.json")
    print("\n[DONE] Training complete! Run `python app.py` to start the API.\n")


if __name__ == '__main__':
    train()
