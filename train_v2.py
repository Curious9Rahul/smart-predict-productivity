import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle, os, json
from utils import generate_synthetic_data

# Unified Encoding Mappings
APP_MAP  = {'Instagram': 0, 'WhatsApp': 1, 'YouTube': 2, 'Chrome': 3, 'Gmail': 4, 'LinkedIn': 5, 'Netflix': 6}
TIME_MAP = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
DAY_MAP  = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
CAT_MAP  = {'Social': 0, 'Productivity': 1, 'Entertainment': 2}
PERSONA_MAP = {'Night Binger': 0, 'Morning Worker': 1, 'Social Butterfly': 2, 'Balanced User': 3, 'Notification Addict': 4, 'Weekend Warrior': 5}

def train_unified_model():
    os.makedirs('model/v2', exist_ok=True)

    # 1. Load Data
    if not os.path.exists('dataset/app_usage.csv'):
        generate_synthetic_data(10000) # Higher quality for XGBoost
    df = pd.read_csv('dataset/app_usage.csv')

    # 2. Vectorize Features
    df['prev_enc']  = df['previous_app'].map(APP_MAP)
    df['time_enc']  = df['time_of_day'].map(TIME_MAP)
    df['day_enc']   = df['day_of_week'].map(DAY_MAP)
    df['cat_enc']   = df['app_category'].map(CAT_MAP)
    df['persona_enc'] = df['persona'].map(PERSONA_MAP)
    
    # Target: We want one model that predicts Next App primarily, 
    # but we can also build others or a multi-output one. 
    # For now, let's create the Unified Next-App Predictor.
    target = df['next_app'].map(APP_MAP)

    features = [
        'prev_enc', 'time_enc', 'day_enc', 'cat_enc', 
        'usage_duration', 'battery_level', 'screen_on_time_today', 
        'session_number', 'is_weekend', 'switch_count_last_hour', 
        'notification_triggered'
    ]
    
    X = df[features]
    y = target
    
    # XGBoost Classifier
    print(f"🚀 Training Unified Intelligence Model (XGBoost)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softprob',
        num_class=len(APP_MAP),
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Unified Model Accuracy: {acc:.2%}")
    
    # 3. Save Artifacts
    pickle.dump(model, open('model/v2/unified_model.pkl', 'wb'))
    
    config = {
        'features': features,
        'app_map': APP_MAP,
        'time_map': TIME_MAP,
        'day_map': DAY_MAP,
        'cat_map': CAT_MAP,
        'persona_map': PERSONA_MAP,
        'version': '2.0.0-Unified'
    }
    with open('model/v2/config.json', 'w') as f:
        json.dump(config, f, indent=2)

if __name__ == '__main__':
    train_unified_model()
