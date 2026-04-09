import pandas as pd
import random
import os
from datetime import datetime, timedelta

# ─── Constants ────────────────────────────────────────────────────────────────
APPS = ['WhatsApp', 'Gmail', 'LinkedIn', 'Chrome', 'YouTube', 'Instagram', 'Netflix']

APP_CATEGORIES = {
    'WhatsApp':  'Social',
    'Gmail':     'Productivity',
    'LinkedIn':  'Social',
    'Chrome':    'Productivity',
    'YouTube':   'Entertainment',
    'Instagram': 'Social',
    'Netflix':   'Entertainment',
}

# Notification likelihood per app (apps that trigger notifications often)
NOTIF_PROB = {
    'WhatsApp':  0.75,
    'Gmail':     0.55,
    'Instagram': 0.45,
    'LinkedIn':  0.30,
    'Chrome':    0.10,
    'YouTube':   0.15,
    'Netflix':   0.05,
}

TIME_SLOTS = ['Morning', 'Afternoon', 'Evening', 'Night']
DAYS       = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
FREQS      = ['Low', 'Medium', 'High']

TIME_PREFS = {
    'Morning':   ['WhatsApp', 'Gmail', 'LinkedIn', 'Chrome'],
    'Afternoon': ['Chrome', 'LinkedIn', 'YouTube', 'WhatsApp', 'Gmail'],
    'Evening':   ['Instagram', 'YouTube', 'WhatsApp'],
    'Night':     ['Instagram', 'YouTube', 'Netflix', 'WhatsApp'],
}

# 6 user personas for richer personalization
USER_PERSONAS = {
    'user_001': {'name': 'Night Binger',     'peak': 'Night',     'fav': ['Netflix', 'YouTube', 'Instagram']},
    'user_002': {'name': 'Morning Worker',   'peak': 'Morning',   'fav': ['Gmail', 'LinkedIn', 'Chrome']},
    'user_003': {'name': 'Social Butterfly', 'peak': 'Evening',   'fav': ['Instagram', 'WhatsApp', 'YouTube']},
    'user_004': {'name': 'Balanced User',    'peak': 'Afternoon', 'fav': ['Chrome', 'Gmail', 'YouTube']},
    'user_005': {'name': 'Notification Addict', 'peak': 'Morning','fav': ['WhatsApp', 'Gmail', 'Instagram']},
    'user_006': {'name': 'Weekend Warrior',  'peak': 'Evening',   'fav': ['Netflix', 'YouTube', 'Chrome']},
}


def _get_duration(app: str, battery: int) -> int:
    """Duration drops when battery is low."""
    cat = APP_CATEGORIES[app]
    base = random.randint(5, 45) if cat == 'Social' else \
           random.randint(15, 90) if cat == 'Entertainment' else \
           random.randint(5, 40)
    # Low battery → shorter sessions
    if battery < 20:
        base = max(3, int(base * 0.5))
    return base


def _detect_distraction(window: list) -> dict:
    """Analyse rolling 5-session window for distraction patterns."""
    if len(window) < 3:
        return {'detected': False, 'pattern': 'None'}
    cats  = [APP_CATEGORIES.get(a, 'Social') for a in window]
    total = len(cats)
    soc   = sum(1 for c in cats if c == 'Social')
    ent   = sum(1 for c in cats if c == 'Entertainment')

    if soc / total >= 0.6:
        return {'detected': True, 'pattern': 'Social Loop'}
    if ent / total >= 0.5:
        return {'detected': True, 'pattern': 'Entertainment Binge'}
    if len(set(window)) == total:
        return {'detected': True, 'pattern': 'App Hopping'}
    return {'detected': False, 'pattern': 'None'}


def generate_synthetic_data(num_samples: int = 5000, seed: int = 42):
    """Generate a rich 19-column dataset for the AI Digital Habit Coach."""
    random.seed(seed)
    data       = []
    start_date = datetime.now() - timedelta(days=60)
    samples_per_user = num_samples // len(USER_PERSONAS)

    for user_id, persona in USER_PERSONAS.items():
        current_app    = random.choice(persona['fav'])
        session_window = []
        screen_today   = 0   # cumulative screen time (minutes) for the day
        session_num    = 1   # session counter within a day
        last_day       = None
        battery        = random.randint(40, 100)  # starting battery

        for i in range(samples_per_user):
            ts = start_date + timedelta(minutes=i * 18)  # ~18 min intervals

            # Detect day change → reset daily counters
            current_day = ts.date()
            if last_day != current_day:
                screen_today = 0
                session_num  = 1
                battery      = random.randint(40, 100)
                last_day     = current_day

            # ── Column 6: day_of_week / Column 17: is_weekend ─────────────
            day        = ts.strftime('%A')
            is_weekend = day in ('Saturday', 'Sunday')

            # ── Column 5: time_of_day ──────────────────────────────────────
            hour = ts.hour
            if 5 <= hour < 12:   time_tag = 'Morning'
            elif 12 <= hour < 17: time_tag = 'Afternoon'
            elif 17 <= hour < 21: time_tag = 'Evening'
            else:                 time_tag = 'Night'

            # Bias toward user's peak time
            if random.random() < 0.45:
                time_tag = persona['peak']

            # ── Column 7: usage_frequency ──────────────────────────────────
            freq = random.choice(FREQS)

            # ── Next app selection (persona-driven) ────────────────────────
            r = random.random()
            if r < 0.55:
                next_app = random.choice(persona['fav'])
            elif r < 0.80:
                next_app = random.choice(TIME_PREFS[time_tag])
            else:
                next_app = random.choice(APPS)

            # ── Column 14: battery_level ───────────────────────────────────
            # Battery drains slowly, charges sometimes
            battery = max(5, battery - random.randint(1, 4))
            if random.random() < 0.05:           # 5% chance of charging event
                battery = random.randint(60, 100)

            # ── Column 8: usage_duration ───────────────────────────────────
            duration = _get_duration(next_app, battery)

            # ── Column 15: screen_on_time_today ───────────────────────────
            screen_today += duration

            # ── Column 16: session_number ──────────────────────────────────
            session_num += 1

            # ── Column 18: switch_count_last_hour ──────────────────────────
            # Approximate from window diversity
            switch_count = len(set(session_window[-6:])) if len(session_window) >= 2 else 1

            # ── Column 19: notification_triggered ─────────────────────────
            notif_triggered = random.random() < NOTIF_PROB.get(next_app, 0.2)

            # ── Columns 11–12: distraction ────────────────────────────────
            session_window = (session_window + [next_app])[-5:]
            dist = _detect_distraction(session_window)

            # ── Column 13: productivity_score ─────────────────────────────
            cat = APP_CATEGORIES[next_app]
            base_score = 80 if cat == 'Productivity' else 40 if cat == 'Social' else 30
            # Penalise for late-night usage, high screen time, many switches
            if time_tag == 'Night' and cat in ('Social', 'Entertainment'):
                base_score -= 15
            if screen_today > 180:
                base_score -= 10
            if switch_count > 4:
                base_score -= 10
            if notif_triggered and cat == 'Social':
                base_score -= 5
            # Weekend bonus for productivity
            if is_weekend and cat == 'Productivity':
                base_score += 10
            base_score = max(5, min(100, base_score))

            data.append({
                # ── Core identity ──────────────────────────────────────────
                'user_id':               user_id,                           # 1
                'persona':               persona['name'],                   # 2
                'timestamp':             ts.strftime('%Y-%m-%d %H:%M:%S'), # 3
                # ── Input features ────────────────────────────────────────
                'previous_app':          current_app,                      # 4
                'time_of_day':           time_tag,                         # 5
                'day_of_week':           day,                               # 6
                'usage_frequency':       freq,                              # 7
                'usage_duration':        duration,                          # 8
                'app_category':          cat,                               # 9
                # ── Target ────────────────────────────────────────────────
                'next_app':              next_app,                          # 10
                # ── Distraction layer ─────────────────────────────────────
                'distraction_detected':  dist['detected'],                  # 11
                'distraction_pattern':   dist['pattern'],                   # 12
                # ── Scoring ───────────────────────────────────────────────
                'productivity_score':    base_score,                        # 13
                # ── NEW contextual features ───────────────────────────────
                'battery_level':         battery,                           # 14
                'screen_on_time_today':  screen_today,                     # 15
                'session_number':        session_num,                       # 16
                'is_weekend':            int(is_weekend),                   # 17
                'switch_count_last_hour': switch_count,                    # 18
                'notification_triggered': int(notif_triggered),            # 19
            })

            current_app = next_app

    df = pd.DataFrame(data)
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/app_usage.csv', index=False)
    print(f"✅ Generated {len(df)} rows × {len(df.columns)} columns → dataset/app_usage.csv")
    print(f"   Users: {df['user_id'].nunique()} | Distraction rate: {df['distraction_detected'].mean():.1%}")
    return df


if __name__ == '__main__':
    generate_synthetic_data(5000)
