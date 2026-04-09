from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle, numpy as np, json, datetime as dt

app = Flask(__name__, template_folder='templates')
CORS(app)

# ─── Load Models & Config ─────────────────────────────────────────────────────
rf_model       = pickle.load(open('model/rf_model.pkl',       'rb'))
markov_matrix  = pickle.load(open('model/markov_matrix.pkl',  'rb'))
kmeans_model   = pickle.load(open('model/kmeans_model.pkl',   'rb'))
distract_model = pickle.load(open('model/distraction_model.pkl', 'rb'))

with open('model/user_profiles.json') as f:
    USER_PROFILES = json.load(f)
with open('model/config.json') as f:
    CFG = json.load(f)

# ─── Mappings ─────────────────────────────────────────────────────────────────
APP_MAP  = CFG['app_map']
TIME_MAP = CFG['time_map']
DAY_MAP  = CFG['day_map']
FREQ_MAP = CFG['freq_map']
CAT_MAP  = CFG['cat_map']        # 'Social' -> 0, etc.
REV_APP  = {v: k for k, v in APP_MAP.items()}

# App → category string
APP_CAT = {
    'Instagram': 'Social',    'WhatsApp':  'Social',
    'LinkedIn':  'Social',    'Gmail':     'Productivity',
    'Chrome':    'Productivity','YouTube': 'Entertainment',
    'Netflix':   'Entertainment',
}

CLUSTER_LABEL = {0: 'Focus Pro', 1: 'Social Butterfly', 2: 'Night Binger', 3: 'Balanced User'}

# Smart interventions (category × time)
INTERVENTIONS = {
    'Social': {
        'Morning':   ('📚 Open a learning app',        'Morning is your peak focus time'),
        'Afternoon': ('✅ Check your task list first',  'Stay productive mid-day'),
        'Evening':   ('🧘 Try 5-min meditation',        'Wind down mindfully'),
        'Night':     ('😴 Enable bedtime mode',         'Late scrolling disrupts sleep'),
    },
    'Entertainment': {
        'Morning':   ('☕ Try a podcast / news app',    'More productive start to the day'),
        'Afternoon': ('💼 Review your work tasks',      'Prime productivity hours'),
        'Evening':   ('📖 Read for 10 minutes first',   'Balance entertainment with growth'),
        'Night':     ('🌙 Set a 20-min screen timer',   'Protect your sleep quality'),
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _time_tag(hour: int) -> str:
    if 5 <= hour < 12:   return 'Morning'
    if 12 <= hour < 17:  return 'Afternoon'
    if 17 <= hour < 21:  return 'Evening'
    return 'Night'

def _auto_context():
    """Return real-time time_of_day and day_of_week."""
    now = dt.datetime.now()
    return _time_tag(now.hour), now.strftime('%A')

def _build_rf_features(prev_enc, time_enc, day_enc, freq_enc,
                        duration, battery, screen_time,
                        session_num, is_weekend, switch_count, notif):
    """Build the 11-feature vector expected by RF model."""
    return [[prev_enc, time_enc, day_enc, freq_enc,
             duration, battery, screen_time,
             session_num, is_weekend, switch_count, notif]]

def _build_dist_features(prev_enc, time_enc, day_enc, freq_enc,
                          duration, cat_enc, battery, screen_time,
                          session_num, is_weekend, switch_count, notif):
    """Build the 12-feature vector expected by distraction GBM."""
    return [[prev_enc, time_enc, day_enc, freq_enc,
             duration, cat_enc, battery, screen_time,
             session_num, is_weekend, switch_count, notif]]

def _build_km_features(duration, prev_enc, time_enc, cat_enc,
                        screen_time, switch_count, battery, notif):
    """Build the 8-feature vector expected by K-Means."""
    return [[duration, prev_enc, time_enc, cat_enc,
             screen_time, switch_count, battery, notif]]

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Hybrid RF + Markov prediction using all contextual features.

    Required: previous_app
    Optional (auto-detected if absent): time_of_day, day_of_week,
              usage_frequency, usage_duration, battery_level,
              screen_on_time_today, session_number, is_weekend,
              switch_count_last_hour, notification_triggered
    """
    data = request.get_json()
    app_val = data.get('previous_app')
    if not app_val or app_val not in APP_MAP:
        return jsonify({'error': f'Invalid previous_app: {app_val}'}), 400

    # Auto-detect time / day if not sent
    auto_time, auto_day = _auto_context()
    time_val  = data.get('time_of_day',   auto_time)
    day_val   = data.get('day_of_week',   auto_day)
    freq_val  = data.get('usage_frequency', 'Medium')

    # New contextual features (with sensible defaults)
    duration    = int(data.get('usage_duration',       20))
    battery     = int(data.get('battery_level',        80))
    screen_time = int(data.get('screen_on_time_today', 60))
    session_num = int(data.get('session_number',        3))
    is_weekend  = int(data.get('is_weekend',  1 if auto_day in ('Saturday','Sunday') else 0))
    switches    = int(data.get('switch_count_last_hour', 2))
    notif       = int(data.get('notification_triggered',  0))

    try:
        curr_idx  = APP_MAP[app_val]
        time_enc  = TIME_MAP[time_val]
        day_enc   = DAY_MAP[day_val]
        freq_enc  = FREQ_MAP[freq_val]
    except KeyError as e:
        return jsonify({'error': f'Invalid field: {e}'}), 400

    # ── Hybrid prediction ─────────────────────────────────────────────────
    X_rf   = _build_rf_features(curr_idx, time_enc, day_enc, freq_enc,
                                 duration, battery, screen_time,
                                 session_num, is_weekend, switches, notif)
    rf_prob  = rf_model.predict_proba(X_rf)[0]
    mk_prob  = markov_matrix[curr_idx]
    combined = (rf_prob * 0.65) + (mk_prob * 0.35)
    pred_idx = int(np.argmax(combined))
    pred_app = REV_APP[pred_idx]
    conf     = float(max(combined))
    category = APP_CAT[pred_app]

    # ── Distraction check ─────────────────────────────────────────────────
    cat_enc  = CAT_MAP[category]
    X_dist   = _build_dist_features(curr_idx, time_enc, day_enc, freq_enc,
                                     duration, cat_enc, battery, screen_time,
                                     session_num, is_weekend, switches, notif)
    distracted      = bool(distract_model.predict(X_dist)[0])
    distract_conf   = float(max(distract_model.predict_proba(X_dist)[0]))

    # ── Behavior cluster ──────────────────────────────────────────────────
    X_km      = _build_km_features(duration, curr_idx, time_enc, cat_enc,
                                    screen_time, switches, battery, notif)
    cluster   = int(kmeans_model.predict(X_km)[0])
    behavior  = CLUSTER_LABEL.get(cluster, 'Balanced User')

    # ── Intervention ──────────────────────────────────────────────────────
    iv_action, iv_reason = None, None
    if category in INTERVENTIONS:
        iv_action, iv_reason = INTERVENTIONS[category].get(
            time_val, (None, None))

    # Extra warnings from new features
    warnings = []
    if screen_time > 180:
        warnings.append(f'📵 {screen_time}m of screen time today — consider a break')
    if battery < 20:
        warnings.append('🔋 Battery critically low — save power')
    if switches > 5:
        warnings.append('🔀 High app-switching detected — focus may be suffering')
    if notif and category == 'Social':
        warnings.append('🔔 Notification-triggered social usage — try Do Not Disturb')

    return jsonify({
        'predicted_app':          pred_app,
        'confidence':             round(conf, 3),
        'category':               category,
        'behavior_type':          behavior,
        'distraction_detected':   distracted,
        'distraction_confidence': round(distract_conf, 2),
        'intervention':           iv_action,
        'warning':                iv_reason,
        'smart_warnings':         warnings,
        'context_used': {
            'time_of_day':     time_val,
            'day_of_week':     day_val,
            'battery_level':   battery,
            'screen_time_min': screen_time,
            'is_weekend':      bool(is_weekend),
            'notif_triggered': bool(notif),
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Full behavioral analysis from a batch of sessions.
    Each session: {app, duration, time, battery, screen_time, notif_triggered, switch_count}
    """
    data     = request.get_json()
    sessions = data.get('sessions', [])
    if not sessions:
        return jsonify({'error': 'No sessions provided'}), 400

    total_dur  = sum(s.get('duration', 10) for s in sessions)
    prod_dur   = sum(s.get('duration', 10) for s in sessions if APP_CAT.get(s.get('app',''), '') == 'Productivity')
    social_dur = sum(s.get('duration', 10) for s in sessions if APP_CAT.get(s.get('app',''), '') == 'Social')
    ent_dur    = total_dur - prod_dur - social_dur

    score  = round(prod_dur / total_dur * 100, 1) if total_dur else 0
    trend  = 'Improving' if score >= 50 else 'Declining'

    # Per-category percentages
    cat_split = {
        'social_pct':        round(social_dur / total_dur * 100, 1) if total_dur else 0,
        'productivity_pct':  round(prod_dur   / total_dur * 100, 1) if total_dur else 0,
        'entertainment_pct': round(ent_dur    / total_dur * 100, 1) if total_dur else 0,
    }

    # Distraction from last session using all features
    last     = sessions[-1]
    prev_enc = APP_MAP.get(last.get('app', 'Instagram'), 0)
    time_enc = TIME_MAP.get(last.get('time', 'Night'), 3)
    cat_enc  = CAT_MAP.get(APP_CAT.get(last.get('app', 'Instagram'), 'Social'), 0)
    dur      = last.get('duration', 20)
    bat      = last.get('battery', 60)
    scr      = last.get('screen_time', 90)
    ses      = last.get('session_number', 3)
    iswk     = last.get('is_weekend', 0)
    sw       = last.get('switch_count', 2)
    ntf      = last.get('notif_triggered', 0)

    X_dist = _build_dist_features(prev_enc, time_enc, 0, 1,
                                   dur, cat_enc, bat, scr, ses, iswk, sw, ntf)
    distracted    = bool(distract_model.predict(X_dist)[0])
    distract_conf = float(max(distract_model.predict_proba(X_dist)[0]))

    X_km  = _build_km_features(dur, prev_enc, time_enc, cat_enc, scr, sw, bat, ntf)
    clust = int(kmeans_model.predict(X_km)[0])
    behavior = CLUSTER_LABEL.get(clust, 'Balanced User')

    dl = 'High' if score < 30 else 'Moderate' if score < 60 else 'Low'

    # Aggregate new-feature stats
    avg_battery   = round(sum(s.get('battery', 80) for s in sessions) / len(sessions), 1)
    total_screen  = sum(s.get('duration', 10) for s in sessions)
    notif_opens   = sum(1 for s in sessions if s.get('notif_triggered', 0))
    avg_switches  = round(sum(s.get('switch_count', 2) for s in sessions) / len(sessions), 1)

    return jsonify({
        'productivity_score':    score,
        'score_trend':           trend,
        'behavior_type':         behavior,
        'distraction_level':     dl,
        'distraction_detected':  distracted,
        'distraction_confidence': round(distract_conf, 2),
        'category_split':        cat_split,
        'avg_battery_level':     avg_battery,
        'total_screen_time_min': total_screen,
        'notification_opens':    notif_opens,
        'avg_switch_count':      avg_switches,
    })


@app.route('/distraction', methods=['POST'])
def check_distraction():
    """Rapid rule-based + ML distraction check from app sequence."""
    data     = request.get_json()
    sequence = data.get('sequence', [])
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    total    = len(sequence)
    social   = sum(1 for a in sequence if APP_CAT.get(a) == 'Social')
    entertain = sum(1 for a in sequence if APP_CAT.get(a) == 'Entertainment')

    if social / total >= 0.6:
        pattern, alert = 'Social Loop', '⚠️ You are in a social media loop! Try stepping away for 10 minutes.'
    elif entertain / total >= 0.5:
        pattern, alert = 'Entertainment Binge', '⚠️ Binge mode detected. Try a short productive task.'
    elif len(set(sequence)) == total:
        pattern, alert = 'App Hopping', '⚠️ Rapid app switching — your focus is fragmented.'
    else:
        pattern, alert = None, None

    # Risk score (0–100)
    risk_score = round((social * 1.5 + entertain) / total * 50 + (len(set(sequence)) / total * 25), 1)
    risk_score = min(100, risk_score)

    return jsonify({
        'distraction_detected': pattern is not None,
        'pattern':              pattern,
        'alert':                alert,
        'social_pct':           round(social / total * 100, 1),
        'entertainment_pct':    round(entertain / total * 100, 1),
        'unique_apps':          len(set(sequence)),
        'risk_score':           risk_score,
    })


@app.route('/productivity', methods=['GET'])
def productivity():
    """Fetch rich per-user profile from personalization engine."""
    uid     = request.args.get('user_id', 'user_001')
    profile = USER_PROFILES.get(uid)
    if not profile:
        return jsonify({'error': f'User {uid} not found'}), 404
    return jsonify(profile)


@app.route('/users', methods=['GET'])
def list_users():
    return jsonify({uid: p['persona'] for uid, p in USER_PROFILES.items()})


@app.route('/stats', methods=['GET'])
def global_stats():
    """Global dataset statistics endpoint."""
    import pandas as pd
    try:
        df = pd.read_csv('dataset/app_usage.csv')
        return jsonify({
            'total_sessions':     int(len(df)),
            'total_users':        int(df['user_id'].nunique()),
            'dataset_columns':    int(len(df.columns)),
            'column_names':       list(df.columns),
            'distraction_rate':   round(df['distraction_detected'].mean() * 100, 1),
            'avg_duration_min':   round(df['usage_duration'].mean(), 1),
            'avg_productivity':   round(df['productivity_score'].mean(), 1),
            'top_app':            df['next_app'].mode()[0],
            'top_distract_pattern': df[df['distraction_detected']]['distraction_pattern'].mode()[0],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
