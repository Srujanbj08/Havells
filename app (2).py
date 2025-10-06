
# Streamlit MVP: Festive Lighting Recommendation Engine
# Run with:  streamlit run app.py
# Files needed: this app.py and catalog.json in the same folder

import json, math
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

CATALOG_FILE = Path(__file__).with_name("catalog.json")

# -----------------------------
# 1) Load catalog
# -----------------------------
@st.cache_data
def load_catalog():
    with open(CATALOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # ensure lists for tags
    df["tags"] = df["tags"].apply(lambda x: x if isinstance(x, list) else [])
    return df

df = load_catalog()

# -----------------------------
# 2) Helpers: normalization, safe divide
# -----------------------------
def minmax(x):
    x = np.asarray(x, dtype=float)
    if np.allclose(x.min(), x.max()):
        # Avoid div-by-zero: return zeros
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())

def safe_div(a, b, eps=1e-9):
    return a / (b + eps)

# -----------------------------
# 3) Aesthetic scoring building blocks (simple, data-driven)
# -----------------------------
SEASON_PALETTES = {
    "Diwali": {"colors": ["warm_white", "gold", "amber", "yellow", "red"]},
    "Christmas": {"colors": ["red", "green", "warm_white", "white"]},
    "Generic": {"colors": ["warm_white", "multicolor", "white"]},
}

STYLE_TAGS = {
    "cozy": ["cozy", "warm_white", "candles", "romantic"],
    "minimal": ["minimal", "modern", "white", "warm_white"],
    "flashy": ["multicolor", "wow", "chasing", "twinkle", "projector"],
    "traditional": ["traditional", "amber", "gold", "red", "green"],
    "modern": ["modern", "accent", "strip"],
}

PATTERN_TAGS = {
    "steady": ["steady"],
    "twinkle": ["twinkle"],
    "chasing": ["chasing"],
    "any": [],
}

def tag_match_score(user_tags, item_tags):
    if not user_tags:
        return 0.0
    # simple ratio of shared tags
    s_user, s_item = set(user_tags), set(item_tags)
    inter = s_user.intersection(s_item)
    return len(inter) / len(s_user)

def color_match_score(season, item_tags):
    palette = SEASON_PALETTES.get(season, SEASON_PALETTES["Generic"])["colors"]
    return 1.0 if any(c in item_tags for c in palette) else 0.0

def pattern_match_score(pattern_pref, item_tags):
    if pattern_pref == "any":
        return 0.5  # neutral bonus
    needed = set(PATTERN_TAGS.get(pattern_pref, []))
    return 1.0 if any(t in item_tags for t in needed) else 0.0

def popularity_score(pop):
    # normalize later; here just pass-through
    return float(pop)

def compute_aesthetic(df, season, style_pref, pattern_pref):
    # Build user tag intent from style
    user_style_tags = STYLE_TAGS.get(style_pref, [])
    tag_sim = df["tags"].apply(lambda t: tag_match_score(user_style_tags, t)).astype(float)
    color_sim = df["tags"].apply(lambda t: color_match_score(season, t)).astype(float)
    pattern_sim = df["tags"].apply(lambda t: pattern_match_score(pattern_pref, t)).astype(float)
    pop = df["popularity"].astype(float)

    # Normalize each to 0..1
    tag_sim_n = minmax(tag_sim)
    color_sim_n = color_sim  # already 0/1
    pattern_sim_n = pattern_sim  # already 0/1
    pop_n = minmax(pop)

    # Weighted sum (tweakable)
    w1, w2, w3, w4 = 0.5, 0.2, 0.15, 0.15
    aesthetic = w1*tag_sim_n + w2*color_sim_n + w3*pattern_sim_n + w4*pop_n
    # Normalize final to 0..1
    aesthetic_n = minmax(aesthetic)
    return aesthetic_n

# -----------------------------
# 4) Energy & cost model
# -----------------------------
def monthly_energy_kwh(watt, hours_per_day=6, days=30):
    return (watt / 1000.0) * hours_per_day * days

def monthly_energy_cost_inr(watt, hours_per_day, price_per_kwh, days=30):
    return monthly_energy_kwh(watt, hours_per_day, days) * price_per_kwh

def compute_energy_norm(df, hours_per_day, price_per_kwh):
    energy_costs = df["watt"].apply(lambda w: monthly_energy_cost_inr(w, hours_per_day, price_per_kwh)).astype(float)
    energy_norm = minmax(energy_costs)  # higher = more expensive
    return energy_costs, energy_norm

# -----------------------------
# 5) Final score & selection (greedy)
# -----------------------------
def compute_final_scores(df, aesthetic_n, energy_norm, aesthetic_weight):
    # A in [0,1]; B = 1-A
    A = float(aesthetic_weight)
    B = 1.0 - A
    final = A*aesthetic_n - B*energy_norm
    # normalize optional (for display) — but selection uses ratio below
    final_n = minmax(final)
    return final, final_n

def greedy_select(df, final_scores, budget_inr, allow_duplicates=False, top_k=5):
    # value-to-price ratio heuristic
    # Avoid division by zero for free items (none in our catalog)
    ratios = final_scores / (df["price_inr"].astype(float) + 1e-6)
    order = np.argsort(-ratios)  # descending
    chosen = []
    spent = 0.0

    for idx in order:
        price = float(df.iloc[idx]["price_inr"])
        if spent + price <= budget_inr:
            chosen.append(int(idx))
            spent += price
        if len(chosen) >= top_k:
            break

    # Fallback: if nothing fits, pick the cheapest positive-score item
    if not chosen:
        positive = [i for i in order if final_scores[i] > 0]
        if positive:
            # pick the cheapest among positive
            cheapest_idx = int(min(positive, key=lambda i: df.iloc[i]["price_inr"]))
            if df.iloc[cheapest_idx]["price_inr"] <= budget_inr:
                chosen = [cheapest_idx]

    return chosen, spent

# -----------------------------
# 6) Simple layout suggestions (rule-based)
# -----------------------------
def layout_suggestions(selection_types, rooms="1-2 rooms"):
    tips = []
    if "string" in selection_types:
        tips.append("Run string lights along balcony railing or window frames.")
    if "curtain" in selection_types:
        tips.append("Hang curtain lights on main window or behind sofa as a backdrop.")
    if "net" in selection_types:
        tips.append("Drape net lights over windows or bushes (if outdoor).")
    if "icicle" in selection_types:
        tips.append("Use icicle lights on facade edges or balcony top.")
    if "decor" in selection_types:
        tips.append("Place LED diyas/candles on shelves and entrance for accents.")
    if "strip" in selection_types:
        tips.append("Place RGB strip under TV unit or along ceiling coves for glow.")
    if "projector" in selection_types:
        tips.append("Aim projector onto a blank wall for patterns; keep ambient lights low.")
    if "spot" in selection_types:
        tips.append("Use warm spotlights to highlight rangoli or Christmas tree.")
    return tips[:4] or ["Distribute lights evenly; avoid glare; keep reachable for switches."]

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Festive Lighting Recommender", page_icon="✨", layout="centered")
st.title("✨ Festive Lighting Recommendation Engine — MVP")

with st.sidebar:
    st.header("Inputs")
    season = st.selectbox("Season / Festival", ["Diwali", "Christmas", "Generic"])
    style_pref = st.selectbox("Style preference", ["cozy", "minimal", "flashy", "traditional", "modern"])
    pattern_pref = st.selectbox("Pattern preference", ["any", "steady", "twinkle", "chasing"])
    budget = st.number_input("Budget (INR)", min_value=100, max_value=100000, value=1500, step=50)
    hours = st.slider("Hours ON per day", 1, 12, 6)
    price_kwh = st.number_input("Electricity price (INR per kWh)", min_value=1.0, max_value=30.0, value=8.0, step=0.5)
    aesthetic_weight = st.slider("Aesthetic vs Energy (0 = only energy, 1 = only aesthetic)", 0.0, 1.0, 0.6, 0.05)

st.caption("MVP logic: rank items by a combined aesthetic score and energy cost; pick best mix under budget.")

# Compute scores
aesthetic_n = compute_aesthetic(df, season, style_pref, pattern_pref)
energy_costs, energy_norm = compute_energy_norm(df, hours, price_kwh)
final, final_n = compute_final_scores(df, aesthetic_n, energy_norm, aesthetic_weight)

# Selection
chosen_indices, spent = greedy_select(df, final, budget, top_k=5)
selection = df.iloc[chosen_indices].copy()

# Compute totals
total_price = float(selection["price_inr"].sum()) if not selection.empty else 0.0
total_watt = float(selection["watt"].sum()) if not selection.empty else 0.0
total_month_kwh = monthly_energy_kwh(total_watt, hours, days=30)
total_month_cost = total_month_kwh * price_kwh

# Display results
st.subheader("🎯 Recommended Setup")
if selection.empty:
    st.info("No items fit the current budget/preferences. Try increasing budget or adjusting the slider.")
else:
    # Add readable columns
    show = selection[["id","name","type","watt","price_inr","tags"]].reset_index(drop=True)
    # computed columns for transparency
    show["aesthetic_0to1"] = np.round(aesthetic_n[chosen_indices], 3)
    show["energy_norm_0to1"] = np.round(energy_norm[chosen_indices], 3)
    show["final_score"] = np.round(final[chosen_indices], 3)
    st.dataframe(show, hide_index=True)

    st.markdown(f"**Spend:** ₹{int(total_price)} / ₹{int(budget)} &nbsp;&nbsp; | &nbsp;&nbsp; **Total Wattage:** {int(total_watt)} W")
    st.markdown(f"**Estimated Monthly Usage:** {total_month_kwh:.2f} kWh  →  **~₹{total_month_cost:.0f} per month**")

    sel_types = set(selection["type"].tolist())
    st.subheader("📐 Simple Layout Tips")
    for tip in layout_suggestions(sel_types):
        st.write(f"- {tip}")

# Transparency panel
with st.expander("How the score is calculated (MVP)"):
    st.markdown("""
- **Aesthetic score (0..1)** = 0.5×tag_match + 0.2×season_color_match + 0.15×pattern_match + 0.15×popularity (each normalized).
- **Energy score (0..1)** = normalized monthly energy cost (kWh×₹/kWh) — higher = more expensive.
- **Final score** = A×Aesthetic − (1−A)×Energy, where **A** is the slider.
- Items are picked greedily by **(final_score / price)** until budget is used.
""")

st.caption("This is a minimal, explainable baseline. Replace tag_match with image embeddings (CLIP) later for richer aesthetics.")
