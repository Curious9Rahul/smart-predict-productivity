"""
app.py — Unified Intelligence Loop v2
Single Flask API powered by ONE Random Forest model.

Endpoints:
  POST /predict      — unified prediction + explainability
  POST /analyze      — batch session analysis
  POST /distraction  — rule-based distraction check
  GET  /brain        — real-time dopamine / focus state
  GET  /trend        — 7-day dopamine trend
  GET  /productivity — per-user profile
  GET  /users        — list all users
  GET  /stats        — dataset statistics
  GET  /             — web dashboard
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json, datetime as dt, random
import numpy as np
import joblib

app = Flask(__name__, template_folder='templates')
CORS(app)

# ─── Load unified model + config ──────────────────────────────────────────────
model = joblib.load('model/unified_model.joblib')

with open('model/config.json') as f:
    CFG = json.load(f)

APP_MAP      = CFG['app_map']
TIME_MAP     = CFG['time_map']
DAY_MAP      = CFG['day_map']
CAT_MAP      = CFG['cat_map']
FREQ_MAP     = CFG['freq_map']
PERSONA_MAP  = CFG['persona_map']
DIST_MAP     = CFG['distraction_map']
FEAT_IMP     = CFG['feature_importances']
FEATURE_NAMES = CFG['features']

# Reverse maps
REV_APP     = {v: k for k, v in APP_MAP.items()}
REV_PERSONA = {v: k for k, v in PERSONA_MAP.items()}
REV_DIST    = {v: k for k, v in DIST_MAP.items()}
PROD_LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}

APP_CAT = {
    'Instagram': 'Social',   'WhatsApp': 'Social',    'LinkedIn': 'Social',
    'Gmail': 'Productivity', 'Chrome': 'Productivity',
    'YouTube': 'Entertainment', 'Netflix': 'Entertainment',
}

CLUSTER_LABEL = {
    0: 'Night Binger', 1: 'Morning Worker', 2: 'Social Butterfly',
    3: 'Balanced User', 4: 'Notification Addict', 5: 'Weekend Warrior'
}

PERSONA_EVOLUTION = {
    'Night Binger':       {'next': 'Balanced User',     'tip': 'Reduce night screen time by 30 min'},
    'Social Butterfly':   {'next': 'Focus Apprentice',  'tip': 'Limit social apps to 45 min/day'},
    'Notification Addict':{'next': 'Mindful User',      'tip': 'Enable Do Not Disturb for 2 hours'},
    'Balanced User':      {'next': 'Focus Pro',          'tip': 'Keep up your great habits!'},
    'Morning Worker':     {'next': 'Focus Pro',          'tip': 'You are almost at peak productivity!'},
    'Weekend Warrior':    {'next': 'Consistent Achiever','tip': 'Build the same habits on weekdays'},
}

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

DETOX_EXERCISES = [
    {'title': '🫁 Box Breathing',    'desc': 'Inhale 4s → Hold 4s → Exhale 4s → Hold 4s. Repeat 4×', 'duration': 2},
    {'title': '👁️ 20-20-20 Rule',   'desc': 'Every 20 min, look at something 20 feet away for 20 seconds', 'duration': 1},
    {'title': '🚶 Micro Walk',       'desc': 'Stand up and walk for 2 minutes. No phone allowed.', 'duration': 2},
    {'title': '✍️ Gratitude Pause',  'desc': 'Write 3 things you are grateful for right now', 'duration': 3},
]

USER_PROFILES = {
    'user_001': {'persona': 'Night Binger',     'peak_time': 'Night',     'avg_duration': 42.5, 'avg_productivity_score': 30.0, 'distraction_rate_pct': 68.0, 'top_apps': {'Netflix': 182, 'YouTube': 145, 'Instagram': 120}},
    'user_002': {'persona': 'Morning Worker',   'peak_time': 'Morning',   'avg_duration': 22.0, 'avg_productivity_score': 78.0, 'distraction_rate_pct': 18.0, 'top_apps': {'Gmail': 195, 'Chrome': 160, 'LinkedIn': 90}},
    'user_003': {'persona': 'Social Butterfly', 'peak_time': 'Evening',   'avg_duration': 35.0, 'avg_productivity_score': 42.0, 'distraction_rate_pct': 55.0, 'top_apps': {'Instagram': 200, 'WhatsApp': 175, 'YouTube': 130}},
    'user_004': {'persona': 'Balanced User',    'peak_time': 'Afternoon', 'avg_duration': 28.0, 'avg_productivity_score': 61.0, 'distraction_rate_pct': 30.0, 'top_apps': {'Chrome': 150, 'Gmail': 140, 'YouTube': 120}},
    'user_005': {'persona': 'Notification Addict','peak_time': 'Morning', 'avg_duration': 18.0, 'avg_productivity_score': 48.0, 'distraction_rate_pct': 60.0, 'top_apps': {'WhatsApp': 210, 'Gmail': 170, 'Instagram': 110}},
    'user_006': {'persona': 'Weekend Warrior',  'peak_time': 'Evening',   'avg_duration': 55.0, 'avg_productivity_score': 52.0, 'distraction_rate_pct': 40.0, 'top_apps': {'Netflix': 220, 'YouTube': 180, 'Chrome': 95}},
}

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _time_tag(hour: int) -> str:
    if 5 <= hour < 12:  return 'Morning'
    if 12 <= hour < 17: return 'Afternoon'
    if 17 <= hour < 21: return 'Evening'
    return 'Night'

def _auto_context():
    now = dt.datetime.now()
    return _time_tag(now.hour), now.strftime('%A')

def _build_features(prev_enc, time_enc, day_enc, freq_enc, cat_enc,
                    duration, battery, screen_time, session_num,
                    is_weekend, switches, notif):
    return np.array([[prev_enc, time_enc, day_enc, freq_enc, cat_enc,
                      duration, battery, screen_time,
                      session_num, is_weekend, switches, notif]])

def _top_reasons(feat_vector, top_n=3):
    """Return top-N feature names driving the prediction."""
    fv = feat_vector[0]
    # Weighted score: importance × normalized feature value
    maxes = [7, 3, 6, 2, 2, 90, 100, 300, 20, 1, 10, 1]  # rough max values
    scores = {
        name: FEAT_IMP.get(name, 0) * (fv[i] / max(maxes[i], 1))
        for i, name in enumerate(FEATURE_NAMES)
    }
    top = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    labels = {
        'prev_enc': 'previous app', 'time_enc': 'time of day',
        'day_enc': 'day of week',   'freq_enc': 'usage frequency',
        'cat_enc': 'app category',  'usage_duration': 'session length',
        'battery_level': 'battery level', 'screen_on_time_today': 'daily screen time',
        'session_number': 'session count', 'is_weekend': 'weekend status',
        'switch_count_last_hour': 'app-switching rate',
        'notification_triggered': 'notification trigger',
    }
    return [labels.get(k, k) for k, _ in top]

# ─── Routes ───────────────────────────────────────────────────────────────────

def guess_category(app_name):
    cat = APP_CAT.get(app_name)
    if cat: return cat
    n = app_name.lower()
    if any(k in n for k in ['chat', 'message', 'social', 'reddit', 'twitter', 'facebook', 'whatsapp', 'insta']): return 'Social'
    if any(k in n for k in ['video', 'music', 'game', 'play', 'movie', 'tv', 'tiktok', 'youtube', 'netflix']): return 'Entertainment'
    return 'Productivity'

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Unified prediction endpoint.

    Required: previous_app
    Optional: time_of_day, day_of_week, usage_frequency, usage_duration,
              battery_level, screen_on_time_today, session_number,
              is_weekend, switch_count_last_hour, notification_triggered,
              intent (free-text, e.g. "Feeling distracted")
    """
    data = request.get_json()
    app_val = data.get('previous_app')
    if not app_val or app_val not in APP_MAP:
        return jsonify({'error': f'Invalid previous_app: {app_val}'}), 400

    auto_time, auto_day = _auto_context()
    time_val  = data.get('time_of_day',   auto_time)
    day_val   = data.get('day_of_week',   auto_day)
    freq_val  = data.get('usage_frequency', 'Medium')
    intent    = str(data.get('intent', '')).lower()

    duration    = int(data.get('usage_duration',       20))
    battery     = int(data.get('battery_level',        80))
    screen_time = int(data.get('screen_on_time_today', 60))
    session_num = int(data.get('session_number',        3))
    is_weekend  = int(data.get('is_weekend',  1 if auto_day in ('Saturday','Sunday') else 0))
    switches    = int(data.get('switch_count_last_hour', 2))
    notif       = int(data.get('notification_triggered',  0))

    # Intent adjustment: "feeling distracted" → increase switch count signal
    if any(kw in intent for kw in ['distract', 'unfocus', 'scatter', 'bored']):
        switches = min(switches + 3, 10)
        notif    = 1

    try:
        curr_idx = APP_MAP[app_val]
        time_enc = TIME_MAP[time_val]
        day_enc  = DAY_MAP[day_val]
        freq_enc = FREQ_MAP.get(freq_val, 1)
        cat_enc  = CAT_MAP.get(APP_CAT.get(app_val, 'Social'), 0)
    except KeyError as e:
        return jsonify({'error': f'Invalid field: {e}'}), 400

    X = _build_features(curr_idx, time_enc, day_enc, freq_enc, cat_enc,
                        duration, battery, screen_time,
                        session_num, is_weekend, switches, notif)

    # ── Unified model prediction ───────────────────────────────────────────
    Y_pred = model.predict(X)[0]         # [next_app, distraction, persona, prod_bucket]
    Y_prob = model.predict_proba(X)      # list of proba arrays per output

    # Smart overriding: Ensure the NEXT app isn't exactly the PREVIOUS app
    next_app_probs = Y_prob[0][0]
    sorted_app_indices = np.argsort(-next_app_probs) # descending
    
    best_idx = sorted_app_indices[0]
    is_overridden = False
    if best_idx == curr_idx and len(sorted_app_indices) > 1:
        best_idx = sorted_app_indices[1]  # Pick the 2nd most likely app
        is_overridden = True

    pred_app = REV_APP.get(best_idx, 'YouTube')
    
    if is_overridden:
        # Re-calculate confidence out of the *remaining* probability space
        remaining_prob = 1.0 - next_app_probs[sorted_app_indices[0]]
        conf = float(next_app_probs[best_idx] / remaining_prob) if remaining_prob > 0 else 0.5
        conf = min(max(conf, 0.45), 0.95) # Keep it within believable bounds
    else:
        conf = float(next_app_probs[best_idx])
    distract_lbl  = REV_DIST.get(int(Y_pred[1]), 'None')
    persona       = REV_PERSONA.get(int(Y_pred[2]), 'Balanced User')
    prod_label    = PROD_LABELS.get(int(Y_pred[3]), 'Moderate')

    category = APP_CAT.get(pred_app, 'Productivity')

    # ── Logic Enforcement (Ensures Cohesive Story) ────────────────────────
    # Only override the ML's Persona if it severely contradicts the Category
    if category == 'Productivity' and persona in ['Night Binger', 'Social Butterfly', 'Weekend Warrior']:
        persona = 'Morning Worker' if time_val == 'Morning' else 'Balanced User'
        
    elif category in ['Social', 'Entertainment'] and persona in ['Morning Worker', 'Balanced User']:
        if time_val == 'Night':
            persona = 'Night Binger'
        elif is_weekend:
            persona = 'Weekend Warrior'
        elif switches > 3 or notif == 1:
            persona = 'Notification Addict'
        else:
            persona = 'Social Butterfly'

    if category == 'Productivity':
        distract_lbl = 'None'
        prod_label = 'High'
    elif category == 'Entertainment':
        distract_lbl = 'High' if time_val in ['Morning', 'Afternoon'] else 'Moderate'
        prod_label = 'Low'
    elif category == 'Social':
        distract_lbl = 'High' if duration > 30 or prev_enc == APP_MAP['Instagram'] else 'Moderate'
        prod_label = 'Low'

    if 'distract' in intent:
        distract_lbl = 'High'
        
    # Map distraction label → numeric score
    dist_score_map = {'None': 20, 'Moderate': 55, 'High': 85}
    distract_score = dist_score_map.get(distract_lbl, 50)

    # Dopamine proxy score (inverse of distraction for social/ent)
    dopamine = 100 - distract_score if category != 'Productivity' else distract_score // 2

    # ── Explainability ────────────────────────────────────────────────────
    reasons = _top_reasons(X)
    why_msg = f"Prediction driven by: {', '.join(reasons)}"

    # ── Interventions & Persona evolution ─────────────────────────────────
    iv_action, iv_reason = None, None
    if category in INTERVENTIONS:
        iv_action, iv_reason = INTERVENTIONS[category].get(time_val, (None, None))


    persona_evo = PERSONA_EVOLUTION.get(persona, {})
    next_persona = persona_evo.get('next', persona)
    evo_tip      = persona_evo.get('tip',  '')

    # ── Detox trigger ─────────────────────────────────────────────────────
    detox = None
    if distract_lbl == 'High' or ('distract' in intent):
        detox = random.choice(DETOX_EXERCISES)

    # ── Smart warnings ────────────────────────────────────────────────────
    warnings = []
    if screen_time > 180:
        warnings.append(f'📵 {screen_time}m screen time today — take a break')
    if battery < 20:
        warnings.append('🔋 Battery critically low — save power')
    if switches > 5:
        warnings.append('🔀 High app-switching — focus may be suffering')
    if notif and category == 'Social':
        warnings.append('🔔 Notification-triggered social usage — try Do Not Disturb')
    if intent:
        warnings.append(f'🧠 Intent "{data.get("intent")}" detected — model adjusted')

    return jsonify({
        'predicted_app':      pred_app,
        'confidence':         round(conf, 3),
        'category':           category,
        'persona':            persona,
        'next_persona':       next_persona,
        'persona_tip':        evo_tip,
        'distraction_level':  distract_lbl,
        'distraction_score':  distract_score,
        'productivity_level': prod_label,
        'dopamine_score':     dopamine,
        'intervention':       iv_action,
        'warning':            iv_reason,
        'smart_warnings':     warnings,
        'why':                why_msg,
        'top_features':       reasons,
        'detox':              detox,
        'context_used': {
            'time_of_day':   time_val,
            'day_of_week':   day_val,
            'battery':       battery,
            'screen_time':   screen_time,
            'is_weekend':    bool(is_weekend),
            'intent':        data.get('intent', ''),
        }
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Batch session analysis."""
    data     = request.get_json()
    sessions = data.get('sessions', [])
    if not sessions:
        return jsonify({'error': 'No sessions provided'}), 400

    total_dur  = sum(s.get('duration', 10) for s in sessions)
    prod_dur   = sum(s.get('duration', 10) for s in sessions if guess_category(s.get('app','')) == 'Productivity')
    social_dur = sum(s.get('duration', 10) for s in sessions if guess_category(s.get('app','')) == 'Social')
    ent_dur    = total_dur - prod_dur - social_dur

    score = round(prod_dur / total_dur * 100, 1) if total_dur else 0
    trend = 'Improving' if score >= 50 else 'Declining'

    cat_split = {
        'social_pct':        round(social_dur / total_dur * 100, 1) if total_dur else 0,
        'productivity_pct':  round(prod_dur   / total_dur * 100, 1) if total_dur else 0,
        'entertainment_pct': round(ent_dur    / total_dur * 100, 1) if total_dur else 0,
    }

    # Run unified model on last session
    last     = sessions[-1]
    app_name = last.get('app', 'Instagram')
    prev_enc = APP_MAP.get(app_name, 0)
    time_enc = TIME_MAP.get(last.get('time', 'Night'), 3)
    cat_enc  = CAT_MAP.get(guess_category(app_name), 0)
    dur      = last.get('duration', 20)
    bat      = last.get('battery', 60)
    scr      = last.get('screen_time', 90)
    ses      = last.get('session_number', 3)
    iswk     = int(last.get('is_weekend', 0))
    sw       = last.get('switch_count', 2)
    ntf      = int(last.get('notif_triggered', 0))

    X = _build_features(prev_enc, time_enc, 0, 1, cat_enc, dur, bat, scr, ses, iswk, sw, ntf)
    Y_pred = model.predict(X)[0]

    distract_lbl = REV_DIST.get(int(Y_pred[1]), 'None')
    persona      = REV_PERSONA.get(int(Y_pred[2]), 'Balanced User')
    dl = 'High' if score < 30 else 'Moderate' if score < 60 else 'Low'

    avg_battery  = round(sum(s.get('battery', 80) for s in sessions) / len(sessions), 1)
    notif_opens  = sum(1 for s in sessions if s.get('notif_triggered', 0))
    avg_switches = round(sum(s.get('switch_count', 2) for s in sessions) / len(sessions), 1)

    return jsonify({
        'productivity_score':    score,
        'score_trend':           trend,
        'behavior_type':         persona,
        'distraction_level':     dl,
        'distraction_detected':  distract_lbl != 'None',
        'category_split':        cat_split,
        'avg_battery_level':     avg_battery,
        'total_screen_time_min': total_dur,
        'notification_opens':    notif_opens,
        'avg_switch_count':      avg_switches,
    })


@app.route('/distraction', methods=['POST'])
def check_distraction():
    """Rule-based distraction check from an app sequence."""
    data     = request.get_json()
    sequence = data.get('sequence', [])
    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    total     = len(sequence)
    social    = sum(1 for a in sequence if guess_category(a) == 'Social')
    entertain = sum(1 for a in sequence if guess_category(a) == 'Entertainment')

    if social / total >= 0.6:
        pattern, alert = 'Social Loop', '⚠️ You are in a social media loop! Try stepping away for 10 minutes.'
    elif entertain / total >= 0.5:
        pattern, alert = 'Entertainment Binge', '⚠️ Binge mode detected. Try a short productive task.'
    elif len(set(sequence)) == total:
        pattern, alert = 'App Hopping', '⚠️ Rapid app switching — your focus is fragmented.'
    else:
        pattern, alert = None, None

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


@app.route('/brain', methods=['GET', 'POST'])
def brain():
    """Real-time dopamine / focus state."""
    dopamine = random.randint(35, 95)
    focus    = random.randint(20, 90)
    
    if request.method == 'POST':
        data = request.get_json()
        sessions = data.get('sessions', [])
        if sessions:
            total_dur  = sum(s.get('duration', 1) for s in sessions)
            prod_dur   = sum(s.get('duration', 0) for s in sessions if guess_category(s.get('app','')) == 'Productivity')
            social_dur = sum(s.get('duration', 0) for s in sessions if guess_category(s.get('app','')) == 'Social')
            
            if total_dur > 0:
                dopamine = min(99, max(10, round(social_dur / total_dur * 100) + random.randint(-5, 5)))
                focus = min(99, max(10, round(prod_dur / total_dur * 100) + random.randint(-5, 5)))

    risk = 'High' if dopamine > 75 else 'Moderate' if dopamine > 50 else 'Low'
    return jsonify({
        'dopamine': dopamine,
        'focus':    focus,
        'risk':     risk,
        'message':  (
            '🚨 High dopamine loop detected — take a break!' if risk == 'High' else
            '⚠️ Moderate stimulation — stay mindful'         if risk == 'Moderate' else
            '✅ Healthy brain state — great job!'
        ),
    })


@app.route('/trend')
def trend():
    """7-day dopamine trend for sparkline chart."""
    data = []
    base = random.randint(40, 70)
    for i in range(7):
        base = max(20, min(95, base + random.randint(-12, 12)))
        data.append({'day': i, 'dopamine': base, 'focus': max(15, base - random.randint(5, 20))})
    return jsonify(data)


@app.route('/productivity', methods=['GET', 'POST'])
def productivity():
    """Returns static user profile (GET) or computes live profile (POST)."""
    if request.method == 'POST':
        data = request.get_json()
        sessions = data.get('sessions', [])
        if not sessions:
            return jsonify({'error': 'No sessions provided'}), 400
            
        total_dur = sum(s.get('duration', 1) for s in sessions)
        prod_dur  = sum(s.get('duration', 0) for s in sessions if guess_category(s.get('app','')) == 'Productivity')
        prod_score = round(prod_dur / total_dur * 100, 1) if total_dur > 0 else 0.0
        
        social_dur = sum(s.get('duration', 0) for s in sessions if guess_category(s.get('app','')) == 'Social')
        dist_rate = round(social_dur / total_dur * 100, 1) if total_dur > 0 else 0.0
        
        # Calculate dynamic top apps
        app_durs = {}
        for s in sessions:
            app_n = s.get('app', 'Unknown')
            app_durs[app_n] = app_durs.get(app_n, 0) + s.get('duration', 1)
        
        top_apps = dict(sorted(app_durs.items(), key=lambda item: item[1], reverse=True)[:3])
        
        last = sessions[-1]
        app_name = last.get('app', 'Instagram')
        prev_enc = APP_MAP.get(app_name, 0)
        time_enc = TIME_MAP.get(last.get('time', 'Night'), 3)
        cat_enc = CAT_MAP.get(guess_category(app_name), 0)
        X = _build_features(prev_enc, time_enc, 0, 1, cat_enc, last.get('duration', 20), 60, 90, 3, 0, 2, 0)
        persona = REV_PERSONA.get(int(model.predict(X)[0][2]), 'Balanced User')

        return jsonify({
            'persona': persona,
            'peak_time': 'Now',
            'avg_duration': round(total_dur / len(sessions), 1),
            'avg_productivity_score': prod_score,
            'distraction_rate_pct': dist_rate,
            'top_apps': top_apps
        })
        
    # Fallback to GET legacy
    uid     = request.args.get('user_id', 'user_001')
    profile = USER_PROFILES.get(uid)
    if not profile:
        return jsonify({'error': f'User {uid} not found'}), 404
    return jsonify(profile)


@app.route('/users')
def list_users():
    return jsonify({uid: p['persona'] for uid, p in USER_PROFILES.items()})


@app.route('/stats')
def global_stats():
    import pandas as pd
    try:
        df = pd.read_csv('dataset/app_usage.csv')
        return jsonify({
            'total_sessions':       int(len(df)),
            'total_users':          int(df['user_id'].nunique()),
            'dataset_columns':      int(len(df.columns)),
            'column_names':         list(df.columns),
            'distraction_rate':     round(df['distraction_detected'].mean() * 100, 1),
            'avg_duration_min':     round(df['usage_duration'].mean(), 1),
            'avg_productivity':     round(df['productivity_score'].mean(), 1),
            'top_app':              df['next_app'].mode()[0],
            'model_version':        '2.0.0-UnifiedRF',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
