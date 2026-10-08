"""
פורטל הערכת שווי נדל"ן - Streamlit app (Hebrew, RTL)
Run:  streamlit run app.py
"""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.house_model import (CATEGORICAL, REFERENCE_YEAR, SAMPLE_HOUSE,
                             predict_with_explanation, train_model)

st.set_page_config(page_title="הערכת שווי נכס | House Price Prediction",
                   page_icon="🏡", layout="wide", initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------- #
# Hebrew labels
# --------------------------------------------------------------------------- #
FEATURE_HE = {
    "sqft_living": "שטח מגורים", "house_age": "גיל הבניין", "bedrooms": "חדרי שינה",
    "bathrooms": "חדרי רחצה", "view": "איכות הנוף", "waterfront": "חזית לים/אגם",
    "floors": "מספר קומות", "condition": "מצב הנכס", "sqft_basement": "שטח מרתף",
    "sqft_lot": "שטח מגרש",
    "waterfront_1": "חזית למים", "view_1": "נוף – דרגה 1", "view_2": "נוף – דרגה 2",
    "view_3": "נוף – דרגה 3", "view_4": "נוף – דרגה 4 (מרהיב)",
}
UNIT_HE = {
    "sqft_living": "לכל רגל רבוע", "sqft_basement": "לכל רגל רבוע", "sqft_lot": "לכל רגל רבוע",
    "house_age": "לכל שנת גיל", "bedrooms": "לכל חדר", "bathrooms": "לכל חדר רחצה",
    "floors": "לכל קומה", "condition": "לכל דרגת מצב",
}
CONDITION_HE = {1: "1 – גרוע", 2: "2 – טעון שיפוץ", 3: "3 – סביר", 4: "4 – טוב", 5: "5 – מצוין"}
VIEW_HE = {0: "0 – ללא נוף", 1: "1 – נוף חלקי", 2: "2 – נוף טוב", 3: "3 – נוף טוב מאוד", 4: "4 – נוף מרהיב"}


LRI, PDI = chr(0x2066), chr(0x2069)   # Unicode left-to-right isolate: keeps "$" and +/- in place inside RTL text


def usd(v: float) -> str:
    return f"{LRI}${v:,.0f}{PDI}"


def usd_range(low: float, high: float) -> str:
    return f"{LRI}${low:,.0f} – ${high:,.0f}{PDI}"


def signed_usd(v: float) -> str:
    return f"{LRI}{'+' if v >= 0 else '−'}${abs(v):,.0f}{PDI}"


def html(markup: str) -> None:
    """Render raw HTML; strips indentation/blank lines so Markdown never turns it into a code block."""
    st.markdown("\n".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Styling: RTL + modern real-estate portal look + responsive
# --------------------------------------------------------------------------- #
html("""
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;800&display=swap" rel="stylesheet">
<style>
:root{
  --brand:#006aff; --brand-dark:#0047b3; --accent:#ff6b35; --ink:#1b2430; --muted:#5f6b7a;
  --card:#ffffff; --bg:#f4f6fa; --line:#e6eaf0; --good:#0a8f5a; --bad:#d64545;
  --shadow:0 4px 18px rgba(16,30,54,.08); --radius:16px;
}
html, body, [class*="css"], .stApp, .stMarkdown, p, li, label, input, button, h1,h2,h3,h4,h5,h6{
  font-family:'Heebo', sans-serif !important;
}
.stApp{ background:var(--bg); }
.stApp, .main, .block-container, [data-testid="stSidebar"]{ direction:rtl; text-align:right; }
.block-container{ padding-top:1.2rem; padding-bottom:1rem; max-width:1250px; }
[data-testid="stMarkdownContainer"] *{ text-align:right; }
[data-testid="stMarkdownContainer"] ul, [data-testid="stMarkdownContainer"] ol{ padding-right:1.3rem; padding-left:0; }
div[data-baseweb="select"] > div, .stNumberInput input{ direction:rtl; text-align:right; }
[data-testid="stSlider"] { direction:ltr; }            /* sliders stay LTR so min→max reads naturally */
[data-testid="stSlider"] label, [data-testid="stWidgetLabel"]{ direction:rtl; text-align:right; width:100%; }
[data-testid="stWidgetLabel"] p{ font-weight:600; color:var(--ink); }
.stTabs [data-baseweb="tab-list"]{ gap:.4rem; direction:rtl; background:#fff; padding:.35rem; border-radius:14px; box-shadow:var(--shadow); }
.stTabs [data-baseweb="tab"]{ height:46px; padding:0 1.2rem; border-radius:10px; font-weight:700; font-size:1rem; }
.stTabs [aria-selected="true"]{ background:var(--brand) !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }
#MainMenu, footer{ visibility:hidden; }

/* hero */
.hero{ background:linear-gradient(120deg,#0b3d91 0%,#006aff 55%,#3fa4ff 100%); color:#fff; border-radius:22px;
  padding:2.2rem 2.4rem; box-shadow:0 12px 30px rgba(0,80,200,.25); position:relative; overflow:hidden; margin-bottom:1.2rem;}
.hero:after{ content:"🏡"; position:absolute; left:2rem; bottom:-1.2rem; font-size:7rem; opacity:.18; }
.hero h1{ color:#fff; font-size:2.3rem; font-weight:800; margin:0 0 .4rem 0; }
.hero p{ color:#e5efff; font-size:1.08rem; margin:0; max-width:720px; }
.chips{ margin-top:1rem; display:flex; gap:.5rem; flex-wrap:wrap; }
.chip{ background:rgba(255,255,255,.16); border:1px solid rgba(255,255,255,.3); padding:.3rem .8rem; border-radius:999px; font-size:.88rem; }

/* cards */
.card{ background:var(--card); border-radius:var(--radius); box-shadow:var(--shadow); padding:1.3rem 1.4rem; border:1px solid var(--line); margin-bottom:1rem; }
.card h3{ margin:0 0 .6rem 0; font-size:1.2rem; color:var(--ink); font-weight:800; }
.section-title{ font-size:1.25rem; font-weight:800; color:var(--ink); margin:.4rem 0 .8rem 0; display:flex; gap:.5rem; align-items:center; }
.price-card{ background:linear-gradient(135deg,#ffffff 0%,#f0f6ff 100%); border:2px solid #cfe2ff; text-align:center !important; }
.price-card *{ text-align:center !important; }
.price-label{ color:var(--muted); font-weight:600; font-size:1rem; }
.price-value{ font-size:3rem; font-weight:800; color:var(--brand-dark); line-height:1.1; margin:.3rem 0; direction:ltr; }
.price-range{ color:var(--muted); font-size:.95rem; direction:rtl; }
.badge{ display:inline-block; padding:.25rem .75rem; border-radius:999px; font-weight:700; font-size:.85rem; margin-top:.6rem; }
.badge-good{ background:#e3f7ee; color:var(--good); } .badge-warn{ background:#fff3e6; color:#b45f06; } .badge-bad{ background:#fde8e8; color:var(--bad); }

.kpis{ display:grid; grid-template-columns:repeat(4,1fr); gap:.8rem; margin-bottom:1rem; }
.kpi{ background:#fff; border-radius:14px; box-shadow:var(--shadow); padding:1rem; border:1px solid var(--line); text-align:center !important; }
.kpi *{ text-align:center !important; }
.kpi .v{ font-size:1.6rem; font-weight:800; color:var(--ink); direction:ltr; }
.kpi .l{ color:var(--muted); font-size:.9rem; font-weight:600; }
.kpi .s{ color:var(--muted); font-size:.78rem; }

.factor{ display:flex; justify-content:space-between; align-items:center; padding:.55rem .2rem; border-bottom:1px dashed var(--line); gap:.6rem; }
.factor:last-child{ border-bottom:none; }
.factor .n{ font-weight:600; color:var(--ink); }
.factor .d{ color:var(--muted); font-size:.85rem; }
.factor .amt{ font-weight:800; direction:ltr; white-space:nowrap; }
.pos{ color:var(--good); } .neg{ color:var(--bad); }

.explain{ background:#fff; border-right:6px solid var(--accent); border-radius:var(--radius); box-shadow:var(--shadow); padding:1.4rem 1.6rem; margin:1rem 0; }
.explain h3{ margin-top:0; color:var(--ink); font-weight:800; }
.explain p, .explain li{ line-height:1.75; color:#2c3644; font-size:1.02rem; }
.ltr{ direction:ltr; unicode-bidi:embed; display:inline-block; }

.formula{ background:#0f1b2d; color:#e8f0ff; border-radius:12px; padding:1rem 1.2rem; direction:ltr; text-align:center !important; font-family:Consolas,monospace !important; font-size:1.02rem; margin:.6rem 0; }

.footer{ margin-top:2.5rem; padding:1.4rem 1rem; text-align:center !important; color:#8a94a3; border-top:1px solid var(--line); font-size:.95rem; }
.footer *{ text-align:center !important; }
.footer b{ color:var(--ink); }

/* responsive */
@media (max-width: 992px){
  .kpis{ grid-template-columns:repeat(2,1fr); }
  .hero h1{ font-size:1.9rem; }
}
@media (max-width: 640px){
  .block-container{ padding-left:.8rem; padding-right:.8rem; }
  .hero{ padding:1.4rem 1.2rem; border-radius:16px; }
  .hero h1{ font-size:1.5rem; } .hero p{ font-size:.95rem; } .hero:after{ font-size:4.5rem; }
  .price-value{ font-size:2.2rem; }
  .kpi .v{ font-size:1.25rem; }
  .stTabs [data-baseweb="tab"]{ padding:0 .6rem; font-size:.9rem; }
  .explain{ padding:1rem 1.1rem; }
}
</style>
""")


# --------------------------------------------------------------------------- #
# Model (trained once and cached)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="מאמן את המודל...")
def get_model():
    return train_model()


bundle = get_model()
m = bundle.metrics
t = m["test"]
data = bundle.data

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
html(f"""
<div class="hero">
  <h1>פורטל הערכת שווי נכסים</h1>
  <p>הערכת מחיר בית מבוססת נתונים, באמצעות מודל <span class="ltr">Linear Regression</span> שאומן על
  {m['n_rows']:,} עסקאות נדל"ן אמיתיות. הזינו את מאפייני הנכס וקבלו הערכת שווי, טווח מחירים והסבר כלכלי מלא.</p>
  <div class="chips">
    <span class="chip">📊 <span class="ltr">R² = {t['r2']:.2f}</span></span>
    <span class="chip">🏘️ {m['n_rows']:,} נכסים</span>
    <span class="chip">⚙️ <span class="ltr">Scikit-learn</span></span>
    <span class="chip">🧠 הסבר סבירות כלכלית</span>
  </div>
</div>
""")

tab_value, tab_model = st.tabs(["🏠  הערכת שווי נכס", "📈  תובנות וביצועי המודל"])

# =========================================================================== #
# TAB 1 – Valuation tool
# =========================================================================== #
with tab_value:
    form_col, result_col = st.columns([5, 7], gap="large")

    with form_col:
        st.markdown('<div class="section-title">📝 מאפייני הנכס</div>', unsafe_allow_html=True)
        with st.container(border=True):
            sqft_living = st.slider("שטח מגורים (רגל רבוע)", 400, 6000, SAMPLE_HOUSE["sqft_living"], 50,
                                    help="1 רגל רבוע ≈ 0.093 מ\"ר")
            st.caption(f"≈ {sqft_living * 0.0929:,.0f} מ\"ר")
            c1, c2 = st.columns(2)
            with c1:
                bedrooms = st.number_input("חדרי שינה", 1, 8, SAMPLE_HOUSE["bedrooms"])
                floors = st.selectbox("מספר קומות", [1, 2, 3], index=SAMPLE_HOUSE["floors"] - 1)
            with c2:
                bathrooms = st.number_input("חדרי רחצה", 1, 6, SAMPLE_HOUSE["bathrooms"])
                yr_built = st.number_input("שנת בנייה", 1900, REFERENCE_YEAR, SAMPLE_HOUSE["yr_built"])
            sqft_basement = st.slider("מתוכו – שטח מרתף (רגל רבוע)", 0, 2500,
                                      SAMPLE_HOUSE["sqft_basement"], 50)
            sqft_lot = st.slider("שטח מגרש (רגל רבוע)", 500, 100000, SAMPLE_HOUSE["sqft_lot"], 500)
            condition = st.select_slider("מצב הנכס", options=list(CONDITION_HE),
                                         value=SAMPLE_HOUSE["condition"], format_func=CONDITION_HE.get)
            view = st.select_slider("איכות הנוף", options=list(VIEW_HE),
                                    value=SAMPLE_HOUSE["view"], format_func=VIEW_HE.get)
            waterfront = st.toggle("🌊 נכס עם חזית למים (ים / אגם)", value=False)
            st.button("💰 חשב הערכת שווי", type="primary", width="stretch")

        if sqft_basement >= sqft_living:
            st.warning("שטח המרתף אינו יכול להיות גדול משטח המגורים הכולל – הוגבל אוטומטית.")
            sqft_basement = int(sqft_living * 0.5)

    house = {
        "bedrooms": bedrooms, "bathrooms": bathrooms, "sqft_living": sqft_living, "sqft_lot": sqft_lot,
        "floors": floors, "waterfront": int(waterfront), "view": view, "condition": condition,
        "sqft_basement": sqft_basement, "yr_built": yr_built,
    }
    r = predict_with_explanation(bundle, house)
    pred = r["prediction"]

    with result_col:
        st.markdown('<div class="section-title">💎 הערכת השווי</div>', unsafe_allow_html=True)

        if pred <= 0:
            st.error("השילוב שנבחר חורג מטווח הנתונים שעליו אומן המודל ולכן התחזית אינה אמינה. "
                     "נסו ערכים מציאותיים יותר (למשל שטח מגורים גדול יותר ביחס למספר החדרים).")
        else:
            sim_med = r["similar_median"]
            gap = (pred - sim_med) / sim_med if sim_med and not np.isnan(sim_med) else 0
            if abs(gap) <= 0.20:
                badge = '<span class="badge badge-good">✔ תואם את מחירי השוק לנכסים דומים</span>'
            elif abs(gap) <= 0.45:
                badge = '<span class="badge badge-warn">⚠ סטייה מתונה ממחירי נכסים דומים – מוסברת במאפייני הנכס</span>'
            else:
                badge = '<span class="badge badge-bad">❗ נכס חריג – ההערכה רחוקה מהחציון לנכסים דומים</span>'

            html(f"""
            <div class="card price-card">
              <div class="price-label">שווי מוערך לנכס</div>
              <div class="price-value">{usd(pred)}</div>
              <div class="price-range">טווח סביר (± <span class="ltr">RMSE</span>):
                 <b class="ltr">{usd_range(r['low'], r['high'])}</b></div>
              {badge}
            </div>
            <div class="kpis">
              <div class="kpi"><div class="l">מחיר לרגל רבוע</div><div class="v">{usd(r['price_per_sqft'])}</div>
                   <div class="s">חציון שוק: {usd(r['market_price_per_sqft'])}</div></div>
              <div class="kpi"><div class="l">חציון נכסים דומים</div><div class="v">{usd(sim_med)}</div>
                   <div class="s">{r['similar_count']:,} נכסים בשטח ±15%</div></div>
              <div class="kpi"><div class="l">בית ממוצע במאגר</div><div class="v">{usd(r['baseline'])}</div>
                   <div class="s">נקודת הייחוס של המודל</div></div>
              <div class="kpi"><div class="l">הפער מהממוצע</div>
                   <div class="v {'pos' if pred >= r['baseline'] else 'neg'}">{signed_usd(pred - r['baseline'])}</div>
                   <div class="s">סכום תרומות המאפיינים</div></div>
            </div>
            """)

            # Contribution waterfall-style bar chart
            contrib = r["contributions"]
            cdf = pd.DataFrame({"מאפיין": [FEATURE_HE.get(k, k) for k in contrib.index],
                                "תרומה": contrib.values})
            cdf["כיוון"] = np.where(cdf["תרומה"] >= 0, "מעלה מחיר", "מוריד מחיר")
            chart = (alt.Chart(cdf).mark_bar(cornerRadius=6, height=18)
                     .encode(x=alt.X("תרומה:Q", title="תרומה למחיר ביחס לבית ממוצע ($)",
                                     axis=alt.Axis(format="$,.0f")),
                             y=alt.Y("מאפיין:N", sort=None, title=None, axis=alt.Axis(orient="right")),
                             color=alt.Color("כיוון:N", scale=alt.Scale(domain=["מעלה מחיר", "מוריד מחיר"],
                                                                        range=["#0a8f5a", "#d64545"]),
                                             legend=alt.Legend(orient="top", title=None)),
                             tooltip=["מאפיין", alt.Tooltip("תרומה:Q", format="$,.0f")])
                     .properties(height=330))
            with st.container(border=True):
                st.markdown("**📊 מה מזיז את המחיר? – פירוק התחזית לפי מאפיינים**")
                st.altair_chart(chart, width="stretch")

    # ------------------------------------------------------------------- #
    # Economic & Reasonability Explanation (full width)
    # ------------------------------------------------------------------- #
    if pred > 0:
        coefs = bundle.coefficients.set_index("feature")["coef_real"]
        contrib = r["contributions"]
        top_up = contrib[contrib > 0].head(3)
        top_down = contrib[contrib < 0].head(3)
        avg = bundle.train_means

        def drivers_html(series):
            return "".join(f"<li><b>{FEATURE_HE.get(k, k)}</b>: <span class='ltr'>{signed_usd(v)}</span></li>"
                           for k, v in series.items()) or "<li>אין</li>"

        bullets = [
            f"<b>שטח המגורים הוא מנוע המחיר המרכזי.</b> לפי המודל, כל רגל רבוע נוסף מוסיף כ-"
            f"<span class='ltr'>{usd(coefs['sqft_living'])}</span> לשווי הנכס. לנכס זה "
            f"<span class='ltr'>{sqft_living:,}</span> ר\"ר לעומת ממוצע של "
            f"<span class='ltr'>{avg['sqft_living']:,.0f}</span> ר\"ר – ולכן השטח לבדו "
            f"{'מעלה' if contrib['sqft_living'] >= 0 else 'מוריד'} את המחיר ב-"
            f"<span class='ltr'>{usd(abs(contrib['sqft_living']))}</span>. זהו ההיגיון הבסיסי בשוק הנדל\"ן: מחיר משקף בעיקר כמות שטח שמיש.",
        ]
        if waterfront:
            bullets.append(
                f"<b>חזית למים היא נכס נדיר.</b> רק כ-1% מהנכסים במאגר נמצאים על קו המים, והמודל מעריך את הפרמיה בכ-"
                f"<span class='ltr'>{usd(coefs['waterfront_1'])}</span>. ההיצע המוגבל והביקוש הגבוה מצדיקים פרמיה זו כלכלית.")
        else:
            bullets.append("<b>ללא חזית למים.</b> הנכס אינו נהנה מפרמיית קו-המים (מאפיין נדיר שמוסיף מאות אלפי דולרים), "
                           "ולכן מחירו נשען על שטח ומאפיינים פנימיים בלבד.")
        if view > 0:
            bullets.append(f"<b>נוף ({VIEW_HE[view]}).</b> נוף פתוח הוא מוצר מבוקש שאי אפשר לשכפל; המודל מוסיף "
                           f"<span class='ltr'>{usd(coefs[f'view_{view}'])}</span> ביחס לנכס ללא נוף.")
        else:
            bullets.append("<b>ללא נוף מיוחד.</b> בדומה לרוב הנכסים במאגר (כ-90%), אין כאן פרמיית נוף.")
        bullets.append(
            f"<b>חדרי שינה מול שטח.</b> בהינתן שטח קבוע, המקדם של חדרי שינה שלילי "
            f"(<span class='ltr'>{usd(coefs['bedrooms'])}</span> לחדר): יותר חדרים באותו שטח = חדרים קטנים וצפופים יותר. "
            f"לעומת זאת חדר רחצה נוסף מוסיף כ-<span class='ltr'>{usd(coefs['bathrooms'])}</span> – סממן של רמת גימור ונוחות.")
        bullets.append(
            f"<b>גיל ומצב.</b> בית בן {REFERENCE_YEAR - yr_built} שנים. במאגר זה, בתים ותיקים נמצאים לרוב בשכונות ותיקות "
            f"ומרכזיות יותר, ולכן המקדם לגיל חיובי (<span class='ltr'>{usd(coefs['house_age'])}</span> לשנה) – "
            f"השפעה עקיפה של מיקום. מצב הנכס ({CONDITION_HE[condition]}) מוסיף כ-"
            f"<span class='ltr'>{usd(coefs['condition'])}</span> לכל דרגה.")

        sim_txt = (f"נכסים במאגר בגודל דומה (±15%) נמכרו בחציון של <span class='ltr'>{usd(r['similar_median'])}</span>, "
                   f"כשמחצית מהם בטווח <span class='ltr'>{usd_range(r['similar_p25'], r['similar_p75'])}</span>. ")
        in_iqr = r["similar_p25"] <= pred <= r["similar_p75"]
        verdict = ("ההערכה נופלת <b>בתוך</b> הטווח הבין-רבעוני של נכסים דומים, ולכן היא סבירה כלכלית ותואמת את השוק."
                   if in_iqr else
                   "ההערכה נמצאת <b>מחוץ</b> לטווח הבין-רבעוני של נכסים דומים – וזה מוסבר במאפיינים הייחודיים של הנכס "
                   "(ראו את הגורמים המובילים למעלה) ולא בהכרח מעיד על טעות.")
        ppsf_gap = (r["price_per_sqft"] / r["market_price_per_sqft"] - 1)

        html(f"""
        <div class="explain">
          <h3>🧠 הסבר סבירות כלכלית</h3>
          <p>המודל מתחיל מ<b>מחיר בית ממוצע במאגר</b> (<span class='ltr'>{usd(r['baseline'])}</span>) ומוסיף או מפחית
          את התרומה של כל מאפיין לפי הסטייה שלו מהממוצע. התוצאה: <b class='ltr'>{usd(pred)}</b>.</p>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:1rem;">
            <div><b class="pos">▲ גורמים שמעלים את המחיר</b><ul>{drivers_html(top_up)}</ul></div>
            <div><b class="neg">▼ גורמים שמורידים את המחיר</b><ul>{drivers_html(top_down)}</ul></div>
          </div>
          <p><b>ניתוח כלכלי לפי מאפיינים:</b></p>
          <ul>{''.join(f'<li>{b}</li>' for b in bullets)}</ul>
          <p><b>בדיקת סבירות מול השוק:</b> {sim_txt}{verdict}
          המחיר לרגל רבוע (<span class='ltr'>{usd(r['price_per_sqft'])}</span>) {'גבוה' if ppsf_gap >= 0 else 'נמוך'} ב-
          <span class='ltr'>{abs(ppsf_gap):.0%}</span> מחציון השוק (<span class='ltr'>{usd(r['market_price_per_sqft'])}</span>).</p>
          <p style="color:#5f6b7a;font-size:.92rem;">⚠️ טווח אי-הוודאות (<span class='ltr'>±{usd(t['rmse'])}</span>) נובע מכך
          שהמודל מסביר כ-<span class='ltr'>{t['r2']:.0%}</span> מהשונות במחירים. גורמים כמו מיקום מדויק (מיקוד/שכונה),
          רמת שיפוץ ותנאי שוק אינם כלולים בנתונים – ולכן יש להתייחס להערכה כנקודת פתיחה מקצועית ולא כשמאות מחייבת.</p>
        </div>
        """)

# =========================================================================== #
# TAB 2 – Model insights & performance
# =========================================================================== #
with tab_model:
    html(f"""
    <div class="kpis">
      <div class="kpi"><div class="l">R² (מקדם הסבר)</div><div class="v">{t['r2']:.3f}</div>
           <div class="s">אימון: {m['train']['r2']:.3f}</div></div>
      <div class="kpi"><div class="l">RMSE</div><div class="v">{usd(t['rmse'])}</div><div class="s">שורש טעות ריבועית ממוצעת</div></div>
      <div class="kpi"><div class="l">MAE</div><div class="v">{usd(t['mae'])}</div><div class="s">טעות מוחלטת ממוצעת</div></div>
      <div class="kpi"><div class="l">MAPE</div><div class="v">{m['mape']:.1%}</div><div class="s">טעות יחסית ממוצעת</div></div>
    </div>
    """)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        html(f"""
        <div class="card">
          <h3>📐 מהי <span class="ltr">Linear Regression</span>?</h3>
          <p>רגרסיה לינארית מניחה שמחיר הנכס הוא <b>סכום משוקלל</b> של מאפייניו. המודל לומד מקדם (<span class="ltr">β</span>)
          לכל מאפיין, כך שסכום ריבועי הטעויות על נתוני האימון יהיה מינימלי (<span class="ltr">OLS</span>).</p>
          <div class="formula">price = β₀ + β₁·sqft_living + β₂·bedrooms + … + βₖ·view_4 + ε</div>
          <p>היתרון המרכזי: <b>שקיפות מלאה</b> – כל מקדם הוא "מחיר צל" של המאפיין, ולכן כל תחזית ניתנת להסבר כלכלי מדויק.</p>
        </div>
        <div class="card">
          <h3>🔧 צינור עיבוד הנתונים</h3>
          <ol>
            <li><b>טעינה וניקוי:</b> {m['n_rows_raw']:,} רשומות, {m['missing_values_raw']} ערכים חסרים. הוסרו כפילויות ומחירים לא תקינים;
                ערכים חסרים מושלמים בחציון (מספרי) או בשכיח (קטגוריאלי).</li>
            <li><b>הנדסת מאפיינים:</b> <span class="ltr">house_age = {REFERENCE_YEAR} − yr_built</span>.
                <span class="ltr">sqft_above</span> הוסר כי הוא שווה בדיוק ל-<span class="ltr">sqft_living − sqft_basement</span> (מולטיקולינאריות מושלמת).</li>
            <li><b>קידוד <span class="ltr">One-Hot</span>:</b> המשתנים <span class="ltr">{', '.join(CATEGORICAL)}</span> קודדו עם
                <span class="ltr">pd.get_dummies(drop_first=True)</span> – למניעת "מלכודת המשתנים הדמי".</li>
            <li><b>חלוקה:</b> <span class="ltr">{m['n_train']:,} / {m['n_test']:,}</span> (80% אימון / 20% בדיקה, <span class="ltr">random_state=42</span>).</li>
            <li><b>נרמול:</b> <span class="ltr">StandardScaler</span> שמותאם על סט האימון בלבד (ללא דליפת מידע) בתוך <span class="ltr">Pipeline</span>.</li>
            <li><b>אימון והערכה:</b> <span class="ltr">LinearRegression</span> ומדדי <span class="ltr">R², RMSE, MAE</span> על סט הבדיקה.</li>
          </ol>
        </div>
        """)

    with c2:
        html(f"""
        <div class="card">
          <h3>📏 איך לקרוא את המדדים?</h3>
          <div class="factor"><div><div class="n"><span class="ltr">R² = {t['r2']:.3f}</span></div>
            <div class="d">המודל מסביר כ-{t['r2']:.0%} מהשונות במחירי הבתים. לנתוני נדל"ן ללא משתני מיקום – תוצאה סבירה.</div></div></div>
          <div class="factor"><div><div class="n"><span class="ltr">RMSE = {usd(t['rmse'])}</span></div>
            <div class="d">טעות "טיפוסית" שמענישה בחומרה טעויות גדולות (בתי יוקרה חריגים).</div></div></div>
          <div class="factor"><div><div class="n"><span class="ltr">MAE = {usd(t['mae'])}</span></div>
            <div class="d">בממוצע התחזית רחוקה בכך מהמחיר בפועל. נמוך מ-<span class="ltr">RMSE</span> ⇐ קיימים מעט נכסים חריגים עם טעות גבוהה.</div></div></div>
          <div class="factor"><div><div class="n">אימון מול בדיקה</div>
            <div class="d"><span class="ltr">R²</span> באימון {m['train']['r2']:.3f} ובבדיקה {t['r2']:.3f} – פער קטן, אין התאמת-יתר משמעותית.</div></div></div>
        </div>
        """)

        # Actual vs predicted
        avp = pd.DataFrame({"מחיר בפועל": bundle.y_test, "מחיר חזוי": bundle.y_pred_test})
        lim = float(max(avp.max()))
        pts = (alt.Chart(avp).mark_circle(size=38, opacity=.45, color="#006aff")
               .encode(x=alt.X("מחיר בפועל:Q", axis=alt.Axis(format="$,.0s")),
                       y=alt.Y("מחיר חזוי:Q", axis=alt.Axis(format="$,.0s")),
                       tooltip=[alt.Tooltip("מחיר בפועל:Q", format="$,.0f"),
                                alt.Tooltip("מחיר חזוי:Q", format="$,.0f")]))
        line = (alt.Chart(pd.DataFrame({"x": [0, lim], "y": [0, lim]}))
                .mark_line(color="#ff6b35", strokeDash=[6, 4]).encode(x="x:Q", y="y:Q"))
        with st.container(border=True):
            st.markdown("**🎯 מחיר בפועל מול חזוי (סט בדיקה)** – ככל שהנקודות קרובות לקו, התחזית מדויקת יותר")
            st.altair_chart((pts + line).properties(height=300), width="stretch")

    # Coefficients / feature importance
    st.markdown('<div class="section-title">🏆 חשיבות מאפיינים וניתוח מקדמים</div>', unsafe_allow_html=True)
    coefs = bundle.coefficients.copy()
    coefs["מאפיין"] = coefs["feature"].map(lambda f: FEATURE_HE.get(f, f))
    coefs["כיוון"] = np.where(coefs["coef_std"] >= 0, "השפעה חיובית", "השפעה שלילית")
    cc1, cc2 = st.columns([6, 5], gap="large")
    with cc1:
        imp = (alt.Chart(coefs).mark_bar(cornerRadius=6, height=16)
               .encode(x=alt.X("coef_std:Q", title="מקדם מתוקנן ($ לסטיית תקן אחת)", axis=alt.Axis(format="$,.0f")),
                       y=alt.Y("מאפיין:N", sort=None, title=None, axis=alt.Axis(orient="right")),
                       color=alt.Color("כיוון:N", scale=alt.Scale(domain=["השפעה חיובית", "השפעה שלילית"],
                                                                  range=["#006aff", "#d64545"]),
                                       legend=alt.Legend(orient="top", title=None)),
                       tooltip=["מאפיין", alt.Tooltip("coef_std:Q", format="$,.0f", title="מתוקנן"),
                                alt.Tooltip("coef_real:Q", format="$,.1f", title="ליחידה")])
               .properties(height=420))
        with st.container(border=True):
            st.markdown("**המקדמים המתוקננים** (אחרי <span class='ltr'>StandardScaler</span>) מאפשרים השוואה הוגנת בין מאפיינים "
                        "ביחידות שונות – ככל שהעמודה ארוכה יותר, המאפיין משפיע יותר.", unsafe_allow_html=True)
            st.altair_chart(imp, width="stretch")
    with cc2:
        rows = "".join(
            f"<div class='factor'><div><div class='n'>{FEATURE_HE.get(f, f)}</div>"
            f"<div class='d'>{UNIT_HE.get(f, 'ביחס לקטגוריית הבסיס')}</div></div>"
            f"<div class='amt {'pos' if v >= 0 else 'neg'}'>{signed_usd(v)}</div></div>"
            for f, v in zip(coefs["feature"], coefs["coef_real"]))
        st.markdown(f"<div class='card'><h3>💲 \"מחיר הצל\" של כל מאפיין</h3>{rows}</div>", unsafe_allow_html=True)

    html(f"""
    <div class="card">
      <h3>💡 תובנות עסקיות מרכזיות</h3>
      <ul>
        <li><b>שטח קובע:</b> שטח המגורים הוא המשתנה החזק ביותר – בערך <span class="ltr">{usd(bundle.coefficients.set_index('feature').at['sqft_living', 'coef_real'])}</span> לכל רגל רבוע.</li>
        <li><b>נדירות = פרמיה:</b> חזית למים ונוף מרהיב מוסיפים מאות אלפי דולרים – מאפיינים שאי אפשר לבנות או לשפץ.</li>
        <li><b>חדרים ≠ ערך:</b> מקדם שלילי לחדרי שינה (בשטח קבוע) מלמד שהשוק מתגמל מרחב ולא חלוקה לחדרים קטנים.</li>
        <li><b>שטח מרתף שווה פחות:</b> המקדם השלילי של המרתף אומר שבאותו שטח כולל, רגל רבוע מתחת לקרקע שווה פחות מרגל רבוע מעליה.</li>
        <li><b>מגבלות:</b> המודל אינו כולל מיקום (מיקוד/קואורדינטות), שהוא בדרך-כלל הגורם החשוב ביותר בנדל"ן. הוספתו, או טרנספורמציה
            לוגריתמית של המחיר, צפויות לשפר משמעותית את <span class="ltr">R²</span>.</li>
      </ul>
    </div>
    """)

# --------------------------------------------------------------------------- #
# Footer
# --------------------------------------------------------------------------- #
html("""
<div class="footer">
  <div><b>פורטל הערכת שווי נכסים</b> · <span class="ltr">Linear Regression · Scikit-learn · Streamlit</span></div>
  <div dir="rtl" style="margin-top:.35rem; direction:rtl; unicode-bidi:isolate;">© כל הזכויות שמורות לאיתי בסטקר</div>
</div>
""")
