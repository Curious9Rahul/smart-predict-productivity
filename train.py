import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import pickle, os, json
from utils import generate_synthetic_data

# ─── Encodings ────────────────────────────────────────────────────────────────
APP_MAP  = {'Instagram': 0, 'WhatsApp': 1, 'YouTube': 2, 'Chrome': 3, 'Gmail': 4, 'LinkedIn': 5, 'Netflix': 6}
TIME_MAP = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
DAY_MAP  = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
FREQ_MAP = {'Low': 0, 'Medium': 1, 'High': 2}
CAT_MAP  = {'Social': 0, 'Productivity': 1, 'Entertainment': 2}


def train_models():
    os.makedirs('model', exist_ok=True)

    # ── 1. Load / Generate ────────────────────────────────────────────────
    if not os.path.exists('dataset/app_usage.csv'):
        generate_synthetic_data(5000)
    df = pd.read_csv('dataset/app_usage.csv')
    print(f"📊 Dataset: {len(df)} rows × {len(df.columns)} cols | {df['user_id'].nunique()} users")

    # ── 2. Encode base features ───────────────────────────────────────────
    df['prev_enc']  = df['previous_app'].map(APP_MAP)
    df['time_enc']  = df['time_of_day'].map(TIME_MAP)
    df['day_enc']   = df['day_of_week'].map(DAY_MAP)
    df['freq_enc']  = df['usage_frequency'].map(FREQ_MAP)
    df['next_enc']  = df['next_app'].map(APP_MAP)
    df['cat_enc']   = df['app_category'].map(CAT_MAP)
    df = df.dropna(subset=['prev_enc', 'time_enc', 'day_enc', 'freq_enc', 'next_enc'])

    # New numeric features (already numeric, just ensure clean)
    num_feats = ['usage_duration', 'battery_level', 'screen_on_time_today',
                 'session_number', 'is_weekend', 'switch_count_last_hour',
                 'notification_triggered']
    for col in num_feats:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # ── 3. Random Forest — Next App Prediction (11 features) ─────────────
    print("\n🤖 Training Random Forest (11 features)...")
    RF_FEATURES = ['prev_enc', 'time_enc', 'day_enc', 'freq_enc',
                   'usage_duration', 'battery_level', 'screen_on_time_today',
                   'session_number', 'is_weekend', 'switch_count_last_hour',
                   'notification_triggered']
    X  = df[RF_FEATURES]
    y  = df['next_enc']
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    rf_acc = accuracy_score(yte, rf.predict(Xte))
    print(f"   RF Accuracy : {rf_acc:.4f}  ({rf_acc:.2%})")
    pickle.dump(rf, open('model/rf_model.pkl', 'wb'))

    # Feature importances
    importances = dict(zip(RF_FEATURES, rf.feature_importances_))
    print("   Top features:", sorted(importances.items(), key=lambda x:-x[1])[:5])

    # ── 4. Markov Chain (persist full matrix) ─────────────────────────────
    print("\n🔗 Building Markov Chain...")
    n = len(APP_MAP)
    mat = np.zeros((n, n))
    for _, row in df.iterrows():
        mat[int(row['prev_enc'])][int(row['next_enc'])] += 1
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    mat /= row_sums
    pickle.dump(mat, open('model/markov_matrix.pkl', 'wb'))
    print("   Saved.")

    # ── 5. K-Means Clustering (8 features) ───────────────────────────────
    print("\n🎯 K-Means Clustering (8 features, 4 clusters)...")
    KM_FEATURES = ['usage_duration', 'prev_enc', 'time_enc', 'cat_enc',
                   'screen_on_time_today', 'switch_count_last_hour',
                   'battery_level', 'notification_triggered']
    cluster_X = df[KM_FEATURES].fillna(0)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=15)
    kmeans.fit(cluster_X)
    pickle.dump(kmeans, open('model/kmeans_model.pkl', 'wb'))
    print("   Saved.")

    # ── 6. Distraction Classifier GBM (12 features) ───────────────────────
    print("\n⚠️  Training Distraction Detector (12 features)...")
    DIST_FEATURES = ['prev_enc', 'time_enc', 'day_enc', 'freq_enc',
                     'usage_duration', 'cat_enc', 'battery_level',
                     'screen_on_time_today', 'session_number', 'is_weekend',
                     'switch_count_last_hour', 'notification_triggered']
    Xd  = df[DIST_FEATURES]
    yd  = df['distraction_detected'].astype(int)
    Xdtr, Xdte, ydtr, ydte = train_test_split(Xd, yd, test_size=0.2, random_state=42)
    gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
    gbm.fit(Xdtr, ydtr)
    dist_acc = accuracy_score(ydte, gbm.predict(Xdte))
    print(f"   Distraction Accuracy: {dist_acc:.4f}  ({dist_acc:.2%})")
    pickle.dump(gbm, open('model/distraction_model.pkl', 'wb'))

    # ── 7. User Profiles (all 19 columns aggregated) ──────────────────────
    print("\n👤 Building rich user profiles...")
    profiles = {}
    for uid, grp in df.groupby('user_id'):
        top_apps       = grp['next_app'].value_counts().head(3).to_dict()
        peak_time      = grp['time_of_day'].mode()[0]
        avg_duration   = round(grp['usage_duration'].mean(), 1)
        avg_prod       = round(grp['productivity_score'].mean(), 1)
        distract_rate  = round(grp['distraction_detected'].mean() * 100, 1)
        avg_battery    = round(grp['battery_level'].mean(), 1)
        avg_screen     = round(grp['screen_on_time_today'].mean(), 1)
        avg_switches   = round(grp['switch_count_last_hour'].mean(), 1)
        notif_rate     = round(grp['notification_triggered'].mean() * 100, 1)
        weekend_score  = round(grp[grp['is_weekend'] == 1]['productivity_score'].mean(), 1) if grp['is_weekend'].sum() > 0 else avg_prod
        weekday_score  = round(grp[grp['is_weekend'] == 0]['productivity_score'].mean(), 1)
        persona        = grp['persona'].iloc[0]
        top_dist_patt  = grp[grp['distraction_detected']]['distraction_pattern'].value_counts().idxmax() \
                         if grp['distraction_detected'].any() else 'None'

        profiles[uid] = {
            'persona':              persona,
            'top_apps':             top_apps,
            'peak_time':            peak_time,
            'avg_duration':         avg_duration,
            'avg_productivity_score': avg_prod,
            'distraction_rate_pct': distract_rate,
            'avg_battery_level':    avg_battery,
            'avg_screen_time_day':  avg_screen,
            'avg_switch_count':     avg_switches,
            'notification_rate_pct': notif_rate,
            'weekend_productivity': weekend_score,
            'weekday_productivity': weekday_score,
            'top_distraction_pattern': top_dist_patt,
        }

    with open('model/user_profiles.json', 'w') as f:
        json.dump(profiles, f, indent=2)
    print(f"   Saved profiles for {len(profiles)} users.")

    # ── 8. Save feature lists for API ─────────────────────────────────────
    meta = {
        'rf_features':   RF_FEATURES,
        'km_features':   KM_FEATURES,
        'dist_features': DIST_FEATURES,
        'apps':          list(APP_MAP.keys()),
        'app_map':       APP_MAP,
        'time_map':      TIME_MAP,
        'day_map':       DAY_MAP,
        'freq_map':      FREQ_MAP,
        'cat_map':       CAT_MAP,
        'categories': {
            'Social':        ['Instagram', 'WhatsApp', 'LinkedIn'],
            'Productivity':  ['Gmail', 'Chrome'],
            'Entertainment': ['YouTube', 'Netflix'],
        }
    }
    with open('model/config.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ All models trained & saved to model/")
    print(f"   RF Accuracy:          {rf_acc:.2%}  (11 features)")
    print(f"   Distraction Accuracy: {dist_acc:.2%} (12 features)")
    print(f"   Dataset columns:      {len(df.columns)}")


if __name__ == '__main__':
    train_models()
