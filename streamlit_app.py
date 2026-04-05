import os
import sys
import html
import base64
import hashlib
import requests
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================================================
# CHATBOT INTEGRATION
# =========================================================
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stratia_chatbot import chatbot_response, get_chat_last_prediction

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="STRATIA",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TABULAR_API_BASE = "http://localhost:5000"
LSTM_API_BASE    = "http://localhost:5001"

### INDICATOR MAPPING (Chat → UI)
INDICATOR_TO_MODEL = {
    "gdp":                   ("Economic Impact Forecasting", "GDP Growth Forecast - LSTM"),
    "inflation":             ("Economic Impact Forecasting", "Inflation - LSTM"),
    "unemployment":          ("Economic Impact Forecasting", "Unemployment - LSTM"),
    "health_expenditure":    ("Health Impact Forecasting",   "Health Forecast - LSTM"),
    "education_expenditure": ("Education Impact Forecasting", "Education Forecast - LSTM"),
    "population":            ("Humanitarian Impact Forecasting", "Population Forecast - ARIMAX"),
    "co2_emissions":         ("Environmental Impact Forecasting", "CO2 - Random Forest"),
    "asylum_seekers":        ("Humanitarian Impact Forecasting", "Asylum seekers - Xgboost"),
    "refugees":              ("Humanitarian Impact Forecasting", "Refugee - PKL"),
}

USERS = {"admin": hashlib.sha256("password123".encode()).hexdigest()}


# =========================================================
# SESSION STATE
# =========================================================
for _k, _v in [("logged_in", False), ("page", "Forecast"),
                ("chat_messages", []), ("last_prediction", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# =========================================================
# HELPERS
# =========================================================
def safe_json(resp):
    try:    return resp.json()
    except: return None

def get_logo_b64():
    try:
        with open("logo.png", "rb") as f:
            return base64.b64encode(f.read()).decode()
    except: return None

def get_api_base(model_name):
    if model_name in ["CO2 - Random Forest", "Refugee - PKL", "Asylum seekers - Xgboost"]:
        return TABULAR_API_BASE
    return LSTM_API_BASE

def build_payload(model_name, country, year, expenditure):
    if model_name == "Population Forecast - ARIMAX":
        return {"model": model_name, "country": country.strip().upper(),
                "year": year, "gdp": expenditure}
    return {"model": model_name, "country": country.strip().upper(),
            "year": year, "expenditure": expenditure}

def format_prediction(value, model_name):
    if model_name == "Population Forecast - ARIMAX":
        return f"{value:,.0f}"
    return f"{value:.4f}"

def _single_predict(api_base, payload):
    try:
        resp = requests.post(f"{api_base}/predict", json=payload, timeout=15)
        if resp.status_code != 200:
            return None
        j = safe_json(resp)
        return float(j["prediction"]) if j and "prediction" in j else None
    except Exception:
        return None

def run_prediction(api_base, payload):
    resp = requests.post(f"{api_base}/predict", json=payload, timeout=20)
    if resp.status_code != 200:
        j = safe_json(resp)
        raise RuntimeError(f"API error ({resp.status_code}): {j['error'] if j and 'error' in j else resp.text}")
    j = safe_json(resp)
    if not j or "prediction" not in j:
        raise RuntimeError("Unexpected API response.")
    return float(j["prediction"])


# =========================================================
# PARALLEL CHART FETCHING  (cached 5 min)
# =========================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_gdp_sensitivity(model_name, country, year, base_expenditure):
    api_base = get_api_base(model_name)
    offsets  = [i * 0.5 for i in range(-10, 11)]
    gdp_vals = [round(max(0.0, base_expenditure + o), 2) for o in offsets]
    labels   = [f"{v:.1f}%" for v in gdp_vals]
    payloads = [build_payload(model_name, country, year, v) for v in gdp_vals]

    values = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=21) as ex:
        futures = {ex.submit(_single_predict, api_base, p): i
                   for i, p in enumerate(payloads)}
        for fut in as_completed(futures):
            i      = futures[fut]
            result = fut.result()
            if result is not None:
                values[i] = round(result, 6)
    return labels, values


@st.cache_data(ttl=300, show_spinner=False)
def fetch_year_trend(model_name, country, base_year, expenditure):
    api_base  = get_api_base(model_name)
    is_arimax = model_name == "Population Forecast - ARIMAX"
    years     = list(range(2025, 2036))
    labels    = [str(y) for y in years]
    payloads  = [
        build_payload(model_name, country,
                      y if is_arimax else base_year, expenditure)
        for y in years
    ]

    values = [None] * len(payloads)
    with ThreadPoolExecutor(max_workers=11) as ex:
        futures = {ex.submit(_single_predict, api_base, p): i
                   for i, p in enumerate(payloads)}
        for fut in as_completed(futures):
            i      = futures[fut]
            result = fut.result()
            if result is not None:
                values[i] = round(result, 6)
    return labels, values


# =========================================================
# CSS
# =========================================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    :root{
        --bg:#060810;--panel:rgba(14,18,32,0.68);--panel-strong:rgba(14,18,32,0.88);
        --panel-soft:rgba(20,24,38,0.72);--input:#141826;
        --border:rgba(255,255,255,0.06);--border-strong:rgba(255,255,255,0.10);
        --text:#e8edf8;--muted:#92a0b8;--muted-2:#627089;
        --accent:#00e5b4;--accent-2:#0090ff;
        --accent-dim:rgba(0,229,180,0.12);--accent-glow:rgba(0,229,180,0.32);
        --danger:#ff4d6a;--radius-xl:24px;--radius-lg:18px;--radius-md:14px;
        --shadow-1:0 20px 60px rgba(0,0,0,0.30);--shadow-2:0 10px 30px rgba(0,0,0,0.22);
        --mono:'IBM Plex Mono',monospace;--sans:'Inter',sans-serif;
    }
    html,body,[data-testid="stAppViewContainer"],[data-testid="stApp"],[data-testid="stMain"]{
        background:var(--bg)!important;color:var(--text)!important;font-family:var(--sans)!important;}
    [data-testid="stAppViewContainer"]::before{content:"";position:fixed;inset:0;
        background:radial-gradient(circle at 12% 18%,rgba(0,229,180,0.08),transparent 28%),
                   radial-gradient(circle at 84% 20%,rgba(0,144,255,0.08),transparent 24%),
                   radial-gradient(circle at 65% 78%,rgba(0,229,180,0.05),transparent 26%);
        pointer-events:none;z-index:0;}
    [data-testid="stAppViewContainer"]::after{content:"";position:fixed;top:0;left:0;right:0;height:3px;
        background:linear-gradient(90deg,transparent,var(--accent),var(--accent-2),transparent);
        z-index:9999;opacity:0.95;}
    [data-testid="block-container"]{max-width:1380px!important;
        padding:1.25rem 1.25rem 2rem!important;position:relative!important;z-index:1!important;}
    #MainMenu,header,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"]{display:none!important;}
    div[data-testid="stHorizontalBlock"]{align-items:stretch!important;}
    [data-testid="stVerticalBlock"]{gap:0.75rem!important;}
    .stratia-topbar{display:flex;align-items:center;justify-content:space-between;gap:16px;
        padding:6px 6px 16px;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:18px;}
    .brand-wrap{display:flex;align-items:center;gap:14px;}
    .brand-title{font-size:1.55rem;font-weight:800;letter-spacing:0.18em;text-transform:uppercase;
        background:linear-gradient(135deg,var(--text),var(--accent));
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1;}
    .brand-sub{font-family:var(--mono);font-size:0.68rem;letter-spacing:0.14em;color:var(--muted);
        text-transform:uppercase;margin-top:6px;line-height:1.4;}
    .admin-pill{font-family:var(--mono);font-size:0.68rem;color:var(--muted);
        border:1px solid var(--border);border-radius:999px;padding:8px 12px;
        background:rgba(255,255,255,0.02);text-transform:uppercase;letter-spacing:0.12em;white-space:nowrap;}
    .section-card{background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.008)),var(--panel-soft);
        border:1px solid var(--border);border-radius:var(--radius-xl);
        padding:22px;box-shadow:var(--shadow-2);height:100%;}
    .micro-label{font-family:var(--mono);font-size:larger;text-transform:uppercase;
        letter-spacing:0.18em;color:white;margin-bottom:25px;line-height:1.4;}
    .hero-title{font-size:2.15rem;line-height:1.08;font-weight:800;letter-spacing:-0.03em;
        margin:0 0 12px;color:var(--text);}
    .hero-sub{color:var(--muted);font-size:0.98rem;line-height:1.7;max-width:760px;margin:0;}
    .context-banner{background:linear-gradient(135deg,rgba(0,229,180,0.08),rgba(0,144,255,0.05));
        border:1px solid rgba(0,229,180,0.16);border-radius:16px;padding:14px 16px;margin-bottom:8px;}
    .context-banner strong{color:var(--accent);}
    .subtle-note{font-size:0.92rem;color:var(--muted);line-height:1.65;margin:0;}
    label,[data-testid="stWidgetLabel"] p{font-family:var(--mono)!important;font-size:0.68rem!important;
        letter-spacing:0.16em!important;text-transform:uppercase!important;
        color:var(--accent)!important;margin-bottom:0.45rem!important;}
    [data-testid="stSelectbox"],[data-testid="stTextInput"],[data-testid="stNumberInput"]{margin-bottom:0.2rem!important;}
    [data-testid="stSelectbox"]>div>div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input{min-height:50px!important;
        background:linear-gradient(180deg,rgba(255,255,255,0.015),rgba(255,255,255,0.005)),var(--input)!important;
        border:1px solid var(--border)!important;border-radius:14px!important;color:var(--text)!important;
        box-shadow:inset 0 1px 0 rgba(255,255,255,0.02)!important;transition:0.25s ease!important;}
    [data-testid="stSelectbox"]>div>div:hover,
    [data-testid="stTextInput"] input:hover,
    [data-testid="stNumberInput"] input:hover{border-color:var(--border-strong)!important;}
    [data-testid="stSelectbox"]>div>div:focus-within,
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus{border-color:var(--accent)!important;
        box-shadow:0 0 0 3px var(--accent-dim),0 0 22px rgba(0,229,180,0.10)!important;}
    [data-testid="stSelectbox"] ul{background:rgba(14,18,32,0.96)!important;
        border:1px solid var(--border)!important;border-radius:16px!important;
        box-shadow:var(--shadow-2)!important;padding:6px!important;}
    [data-testid="stSelectbox"] li{border-radius:10px!important;}
    [data-testid="stSelectbox"] li:hover,
    [data-testid="stSelectbox"] li[aria-selected="true"]{background:var(--accent-dim)!important;color:var(--accent)!important;}
    [data-testid="stButton"]{margin-top:0.1rem!important;}
    [data-testid="stButton"]>button{min-height:48px!important;width:100%!important;
        border-radius:14px!important;border:1px solid rgba(0,229,180,0.36)!important;
        background:linear-gradient(135deg,rgba(255,255,255,0.02),rgba(0,229,180,0.06))!important;
        color:var(--accent)!important;font-weight:700!important;letter-spacing:0.14em!important;
        text-transform:uppercase!important;font-size:0.78rem!important;
        transition:0.25s ease!important;box-shadow:0 8px 24px rgba(0,0,0,0.22)!important;}
    [data-testid="stButton"]>button:hover{transform:translateY(-1px);
        box-shadow:0 0 24px var(--accent-glow),0 14px 30px rgba(0,0,0,0.30)!important;
        border-color:var(--accent)!important;}
    [data-testid="stButton"]>button:active{transform:translateY(0);}
    div[data-testid="stMetric"]{
        background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.005)),var(--panel-strong)!important;
        border:1px solid var(--border)!important;border-radius:20px!important;
        padding:20px!important;box-shadow:var(--shadow-2)!important;}
    [data-testid="stMetricLabel"]{font-family:var(--mono)!important;font-size:0.68rem!important;
        letter-spacing:0.16em!important;text-transform:uppercase!important;color:var(--accent)!important;}
    [data-testid="stMetricValue"]{font-size:2.25rem!important;font-weight:800!important;color:var(--text)!important;}
    div[data-testid="stRadio"]>div{display:flex!important;flex-direction:row!important;
        gap:8px!important;margin-bottom:0!important;align-items:center!important;}
    div[data-testid="stRadio"] label{padding:10px 16px!important;border-radius:12px!important;
        border:1px solid transparent!important;background:transparent!important;
        transition:0.2s ease!important;margin:0!important;}
    div[data-testid="stRadio"] label:hover{background:rgba(255,255,255,0.03)!important;
        border-color:rgba(255,255,255,0.05)!important;}
    div[data-testid="stRadio"] label:has(input:checked){
        background:linear-gradient(135deg,rgba(0,229,180,0.10),rgba(0,144,255,0.07))!important;
        border-color:rgba(0,229,180,0.20)!important;}
    div[data-testid="stRadio"] div[data-baseweb="radio"]>div:first-child{display:none!important;}
    .tiny-caption{font-size:0.84rem;color:var(--muted);line-height:1.6;margin:0;}
    .badge-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:18px;}
    .mini-stat{background:rgba(255,255,255,0.02);border:1px solid var(--border);
        border-radius:16px;padding:14px;}
    .mini-stat .k{font-family:var(--mono);font-size:0.62rem;color:var(--muted);
        text-transform:uppercase;letter-spacing:0.14em;margin-bottom:8px;}
    .mini-stat .v{font-size:1rem;font-weight:700;color:var(--text);line-height:1.45;}
    ::-webkit-scrollbar{width:7px;}
    ::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--accent-2),var(--accent));border-radius:999px;}
    ::-webkit-scrollbar-track{background:transparent;}

    /* ── CHAT STYLING ─────────────────────────────────── */
    /* Style the native Streamlit chat elements */
    [data-testid="stChatMessage"]{
        background:rgba(14,18,32,0.60) !important;
        border:1px solid rgba(255,255,255,0.06) !important;
        border-radius:14px !important;
        padding:10px 14px !important;
        font-size:0.85rem !important;
        line-height:1.7 !important;
    }
    /* User messages slightly blue-tinted */
    [data-testid="stChatMessage"][data-testid*="user"],
    [data-testid="stChatMessageContent"]:has(+ [aria-label="user"]){
        background:rgba(0,144,255,0.06) !important;
        border-color:rgba(0,144,255,0.12) !important;
    }
    [data-testid="stChatInput"]{
        background:rgba(14,18,32,0.80) !important;
        border:1px solid rgba(255,255,255,0.08) !important;
        border-radius:16px !important;
    }
    [data-testid="stChatInput"] textarea{
        background:transparent !important;
        color:var(--text) !important;
        font-family:var(--sans) !important;
        font-size:0.88rem !important;
    }
    /* Chat container wrapper */
    .chat-wrapper{
        background:rgba(10,14,26,0.72);
        border:1px solid rgba(255,255,255,0.07);
        border-radius:20px;
        padding:16px;
    }
    .chat-section-label{
        font-family:var(--mono);
        font-size:0.62rem;
        letter-spacing:0.16em;
        text-transform:uppercase;
        color:#627089;
        padding-bottom:8px;
        border-bottom:1px solid rgba(255,255,255,0.05);
        margin-bottom:10px;
    }
    .powered-tag{
        font-family:var(--mono);
        font-size:0.60rem;
        color:#627089;
        letter-spacing:0.10em;
        text-transform:uppercase;
    }
    /* Quick action buttons — smaller style */
    .stButton.quick-btn > button{
        min-height:36px !important;
        font-size:0.70rem !important;
        padding:6px 10px !important;
        border-radius:10px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================================
# CHART RENDERER
# =========================================================
def render_forecast_chart(p: dict, chart_mode: str):
    model_name  = p["model"]
    country     = p["country"]
    year        = p["year"]
    expenditure = p["expenditure"]
    is_pop      = model_name == "Population Forecast - ARIMAX"

    if chart_mode == "gdp":
        labels, values = fetch_gdp_sensitivity(model_name, country, year, expenditure)
        x_label        = "Defense Expenditure (% GDP)"
        title_suffix   = "GDP Sensitivity  (base ± 5%,  step 0.5%)"
        current_index  = 10
        tick_limit     = 11
    else:
        labels, values = fetch_year_trend(model_name, country, year, expenditure)
        x_label        = "Year"
        title_suffix   = "Year Trend  2025 – 2035"
        clamped        = max(2025, min(2035, year))
        current_index  = clamped - 2025
        tick_limit     = 11

    def jv(v): return "null" if v is None else str(v)

    values_js    = ", ".join(jv(v) for v in values)
    labels_js    = ", ".join(f'"{l}"' for l in labels)
    hl           = ["null"] * len(values)
    if values[current_index] is not None:
        hl[current_index] = str(values[current_index])
    highlight_js = ", ".join(hl)

    y_label     = "Population" if is_pop else "Predicted Value"
    title_esc   = html.escape(
        model_name.replace(" - LSTM","").replace(" - ARIMAX","")
                  .replace(" - Random Forest","").replace(" - Xgboost",""))
    country_esc = html.escape(country)
    tooltip_fmt = (
        "return ' '+c.dataset.label+': '+Number(c.raw).toLocaleString(undefined,{maximumFractionDigits:0});"
        if is_pop else
        "return ' '+c.dataset.label+': '+Number(c.raw).toFixed(4);"
    )

    st.components.v1.html(f"""
    <div style="background:linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.005)),rgba(14,18,32,0.78);
        border:1px solid rgba(255,255,255,0.06);border-radius:22px;
        padding:18px 20px 16px;box-shadow:0 14px 34px rgba(0,0,0,0.24);">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:12px;margin-bottom:12px;">
            <div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;letter-spacing:0.16em;text-transform:uppercase;color:#00e5b4;margin-bottom:5px;">
                    {html.escape(title_suffix)}</div>
                <div style="font-family:'Inter',sans-serif;font-size:1.02rem;font-weight:700;color:#e8edf8;">
                    {title_esc} <span style="color:#627089;">·</span> <span style="color:#25e7bf;">{country_esc}</span></div>
            </div>
            <div style="text-align:right;">
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.60rem;color:#627089;letter-spacing:0.12em;text-transform:uppercase;">Base Expenditure</div>
                <div style="font-size:0.95rem;color:#0090ff;font-weight:700;">{expenditure:.1f}% GDP</div>
            </div>
        </div>
        <canvas id="sc" style="width:100%;max-height:300px;"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script>
    (function(){{
        const ctx = document.getElementById("sc").getContext("2d");
        new Chart(ctx,{{
            type:"line",
            data:{{
                labels:[{labels_js}],
                datasets:[
                    {{label:"Forecast",data:[{values_js}],
                      borderColor:"#00e5b4",borderWidth:2.4,
                      backgroundColor:"rgba(0,229,180,0.07)",fill:true,
                      pointRadius:2,pointHoverRadius:5,
                      pointBackgroundColor:"#00e5b4",
                      tension:0.35,spanGaps:false}},
                    {{label:"Selected",data:[{highlight_js}],
                      borderColor:"transparent",backgroundColor:"#ff9f3f",
                      pointRadius:9,pointHoverRadius:11,
                      pointStyle:"circle",showLine:false}}
                ]
            }},
            options:{{
                responsive:true,maintainAspectRatio:false,
                animation:{{duration:600,easing:"easeInOutQuart"}},
                interaction:{{mode:"index",intersect:false}},
                plugins:{{
                    legend:{{display:false}},
                    tooltip:{{backgroundColor:"rgba(14,18,32,0.95)",
                        borderColor:"rgba(255,255,255,0.08)",borderWidth:1,
                        titleColor:"#00e5b4",bodyColor:"#d6e1f3",padding:12,
                        callbacks:{{label:function(c){{
                            if(c.raw===null||c.dataset.label==="Selected") return null;
                            {tooltip_fmt}
                        }}}}
                    }}
                }},
                scales:{{
                    x:{{grid:{{color:"rgba(255,255,255,0.04)"}},
                        ticks:{{color:"#6e7d92",maxTicksLimit:{tick_limit}}},
                        title:{{display:true,text:"{html.escape(x_label)}",color:"#627089",font:{{size:11}}}}}},
                    y:{{grid:{{color:"rgba(255,255,255,0.04)"}},
                        ticks:{{color:"#6e7d92"}},
                        title:{{display:true,text:"{y_label}",color:"#627089",font:{{size:11}}}}}}
                }}
            }}
        }});
    }})();
    </script>
    """, height=420, scrolling=False)


def render_empty_chart():
    st.markdown("""
    <div style="background:rgba(14,18,32,0.78);border:1px solid rgba(255,255,255,0.06);
        border-radius:22px;padding:60px 20px;text-align:center;">
        <div style="font-family:'IBM Plex Mono',monospace;font-size:0.66rem;letter-spacing:0.16em;
            text-transform:uppercase;color:#00e5b4;margin-bottom:12px;">Forecast Visualization</div>
        <div style="color:#627089;font-size:0.95rem;line-height:1.7;">
            Run a prediction on the Forecast page to populate this chart.</div>
    </div>""", unsafe_allow_html=True)


# =========================================================
# LOGIN
# =========================================================
def login_page():
    logo_b64  = get_logo_b64()
    logo_html = (f'<img src="data:image/png;base64,{logo_b64}" width="44" style="border-radius:12px;">'
                 if logo_b64 else
                 '<div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#00e5b4,#0090ff);"></div>')
    st.markdown(f"""
    <div class="stratia-topbar">
        <div class="brand-wrap">{logo_html}
            <div><div class="brand-title">STRATIA</div>
                 <div class="brand-sub">Strategic Forecasting Interface</div></div>
        </div>
        <div class="admin-pill">Secure Access</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("""
        <div class="section-card" style="min-height:420px;display:flex;flex-direction:column;justify-content:center;">
            <div class="micro-label">Defense · Macro · Geopolitics</div>
            <div class="hero-title">Strategic intelligence with a cleaner operating surface.</div>
            <div class="hero-sub">STRATIA combines forecasting models, comparative geopolitical context,
                and an intelligence assistant into one decision-support interface.</div>
            <div class="badge-grid">
                <div class="mini-stat"><div class="k">Domains</div><div class="v">5 Forecast Areas</div></div>
                <div class="mini-stat"><div class="k">Models</div><div class="v">LSTM · RF · XGBoost · ARIMA</div></div>
                <div class="mini-stat"><div class="k">Mode</div><div class="v">Analyst Console</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="micro-label">Authentication</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.4rem;font-weight:800;margin-bottom:8px;">Enter the platform</div>', unsafe_allow_html=True)
        st.markdown('<div class="subtle-note" style="margin-bottom:18px;">Use the demo credentials to access the forecasting dashboard.</div>', unsafe_allow_html=True)
        username = st.text_input("Username", placeholder="admin")
        password = st.text_input("Password", type="password", placeholder="••••••••••")
        if st.button("Authenticate", use_container_width=True):
            if username in USERS and USERS[username] == hashlib.sha256(password.encode()).hexdigest():
                st.session_state.logged_in = True
                st.success("Access granted.")
                st.rerun()
            else:
                st.error("Invalid credentials.")
        st.caption("Demo login: admin / password123")


# =========================================================
# HEADER + NAV
# =========================================================
def render_app_header():
    logo_b64  = get_logo_b64()
    logo_html = (f'<img src="data:image/png;base64,{logo_b64}" width="34" style="border-radius:10px;">'
                 if logo_b64 else
                 '<div style="width:34px;height:34px;border-radius:10px;background:linear-gradient(135deg,#00e5b4,#0090ff);"></div>')
    st.markdown(f"""
    <div class="stratia-topbar">
        <div class="brand-wrap">{logo_html}
            <div><div class="brand-title">STRATIA</div>
                 <div class="brand-sub">Strategic Intelligent Assistant</div></div>
        </div>
        <div class="admin-pill">admin</div>
    </div>""", unsafe_allow_html=True)

    nav_col, logout_col = st.columns([8, 1], gap="small")
    with nav_col:
        page = st.radio("Nav", ["Forecast", "Intelligence"], horizontal=True,
                        label_visibility="collapsed",
                        index=0 if st.session_state.page == "Forecast" else 1)
        st.session_state.page = page
    with logout_col:
        if st.button("Logout", use_container_width=True):
            for k, v in [("logged_in", False), ("page", "Forecast"),
                         ("chat_messages", []), ("last_prediction", None)]:
                st.session_state[k] = v
            st.rerun()


# =========================================================
# FORECAST PAGE
# =========================================================
def forecast_page():
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="micro-label">Model Configuration</div>', unsafe_allow_html=True)
        model_choice = st.selectbox("Prediction Domain", [
            "Economic Impact Forecasting", "Education Impact Forecasting",
            "Health Impact Forecasting", "Environmental Impact Forecasting",
            "Humanitarian Impact Forecasting",
        ])
        model_map = {
            "Economic Impact Forecasting":      ["GDP Growth Forecast - LSTM", "Inflation - LSTM", "Unemployment - LSTM"],
            "Education Impact Forecasting":     ["Education Forecast - LSTM"],
            "Health Impact Forecasting":        ["Health Forecast - LSTM"],
            "Environmental Impact Forecasting": ["CO2 - Random Forest"],
            "Humanitarian Impact Forecasting":  ["Population Forecast - ARIMAX", "Refugee - PKL", "Asylum seekers - Xgboost"],
        }

        # Strip technical suffixes for display only
        DISPLAY_LABELS = {
            "GDP Growth Forecast - LSTM":   "GDP Growth Forecast",
            "Inflation - LSTM":             "Inflation",
            "Unemployment - LSTM":          "Unemployment",
            "Education Forecast - LSTM":    "Education Forecast",
            "Health Forecast - LSTM":       "Health Forecast",
            "CO2 - Random Forest":          "CO2 Emissions",
            "Population Forecast - ARIMAX": "Population Forecast",
            "Refugee - PKL":                "Refugee",
            "Asylum seekers - Xgboost":     "Asylum Seekers",
        }

        internal_models = model_map[model_choice]
        display_models  = [DISPLAY_LABELS[m] for m in internal_models]

        selected_display = st.selectbox("Prediction Model", display_models)
        sub_model_choice = internal_models[display_models.index(selected_display)]
        api_base = get_api_base(sub_model_choice)

        c1, c2 = st.columns(2)
        with c1: country     = st.selectbox("Country", ["USA", "CHN", "RUS", "GBR", "FRA"], index=2)
        with c2: expenditure = st.number_input("Defense Expenditure (% of GDP)",
                                               min_value=0.0, max_value=100.0,
                                               value=8.0, step=0.1, format="%.1f")
        year = st.number_input("Year", min_value=1960, max_value=2100, value=2025, step=1)

        payload = build_payload(sub_model_choice, country, int(year), float(expenditure))

        if st.button("Run Prediction", use_container_width=True):
            try:
                with st.spinner("Processing forecast..."):
                    pred = run_prediction(api_base, payload)
                st.session_state.last_prediction = {
                    "value": pred, "model": sub_model_choice, "country": country,
                    "year": int(year), "expenditure": float(expenditure), "domain": model_choice,
                }
                fetch_gdp_sensitivity.clear()
                fetch_year_trend.clear()
                st.success("Prediction complete.")
                st.rerun()
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot connect to API.\n- Tabular: {TABULAR_API_BASE}\n- LSTM: {LSTM_API_BASE}")
            except requests.exceptions.Timeout:
                st.error("Request timed out.")
            except Exception as e:
                st.error(str(e))

    with right:
        st.markdown('<div class="micro-label">Latest Output</div>', unsafe_allow_html=True)
        p = st.session_state.last_prediction
        if p:
            st.metric("Forecast Output", format_prediction(p["value"], p["model"]))
            st.markdown(f"""
            <div class="context-banner"><div class="subtle-note">
                <strong>{html.escape(p['model'])}</strong><br>
                Country: {html.escape(p['country'])}<br>
                Year: {p['year']}<br>
                Defense Expenditure: {p['expenditure']}% of GDP<br>
                Domain: {html.escape(p['domain'])}
            </div></div>""", unsafe_allow_html=True)
            st.markdown('<div class="tiny-caption">Result is available as context in the Intelligence page.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="subtle-note">No prediction yet. Configure a model and run a scenario.</div>', unsafe_allow_html=True)


# =========================================================
# CHAT PANEL  — uses native st.chat_message + st.chat_input
# =========================================================
def render_chat_panel():
    # ── Header label ─────────────────────────────────────
    st.markdown("""
    <div style="display:flex;justify-content:space-between;align-items:center;
        padding-bottom:8px;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:12px;">
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.62rem;
            letter-spacing:0.16em;text-transform:uppercase;color:#627089;">
            Intelligence Assistant
        </span>
        <span style="font-family:'IBM Plex Mono',monospace;font-size:0.60rem;
            color:#627089;letter-spacing:0.10em;text-transform:uppercase;">
            Groq · Llama 3.3 70B
        </span>
    </div>""", unsafe_allow_html=True)

    # ── Quick action buttons ──────────────────────────────
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    quick_prompts = {
        "GDP · Russia 2030":      "predict gdp for russia in 2030 defense spending 2.5%",
        "Inflation · USA 2028":   "predict inflation for america in 2028 defense spending 3.2%",
        "CO2 · China 2027":       "predict co2 emissions for china in 2027 defense spending 1.8%",
        "Refugees · France 2029": "predict refugees for france in 2029 defense spending 2.1%",
    }
    cols = [r1c1, r1c2, r2c1, r2c2]
    triggered_prompt = None
    for col, (label, prompt) in zip(cols, quick_prompts.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"quick_{label}"):
                triggered_prompt = prompt

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    # ── Message history with fixed height ─────────────────
    with st.container(height=435, border=False):
        msgs = st.session_state.chat_messages
        if not msgs:
            st.markdown("""
            <div style="text-align:center;padding:32px 12px;
                background:rgba(4,6,14,0.50);border:1px solid rgba(255,255,255,0.05);
                border-radius:14px;margin-bottom:10px;">
                <div style="font-size:1.6rem;margin-bottom:8px;">🧠</div>
                <div style="font-size:0.85rem;color:#627089;line-height:1.7;">
                    Ask STRATIA a natural language question<br>to get a full forecast + AI insight.
                </div>
                <div style="font-family:'IBM Plex Mono',monospace;font-size:0.65rem;color:#3d4d60;
                    background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);
                    border-radius:8px;padding:6px 12px;display:inline-block;margin-top:10px;">
                    predict [indicator] for [country] in [year] defense [X]%
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            for m in msgs:
                role = "user" if m["role"] == "user" else "assistant"
                with st.chat_message(role):
                    st.markdown(m["content"])

    # ── Chat input ────────────────────────────────────────
    user_input = st.chat_input(
        "Ask STRATIA — predict gdp for china in 2028 military spending 2%...",
        key="stratia_chat_input",
    )

    # ── Handle quick button OR typed input ────────────────
    final_input = triggered_prompt or (user_input.strip() if user_input else None)

    if final_input:
        st.session_state.chat_messages.append({"role": "user", "content": final_input})
        with st.chat_message("user"):
            st.markdown(final_input)
        with st.chat_message("assistant"):
            with st.spinner("Running forecast..."):
                try:
                    # Sync start: record current chat prediction state
                    prev_chat_last = get_chat_last_prediction().copy()
                    
                    reply = chatbot_response(final_input)
                    
                    # Sync end: check if a new prediction was made
                    curr_chat_last = get_chat_last_prediction()
                    if curr_chat_last["prediction"] is not None and curr_chat_last != prev_chat_last:
                        ind = curr_chat_last["indicator"]
                        if ind in INDICATOR_TO_MODEL:
                            dom, mod = INDICATOR_TO_MODEL[ind]
                            st.session_state.last_prediction = {
                                "value":       curr_chat_last["prediction"],
                                "model":       mod,
                                "country":     curr_chat_last["country"],
                                "year":        int(curr_chat_last["year"]),
                                "expenditure": float(curr_chat_last["defense"]),
                                "domain":      dom,
                            }
                            # Clear chart cache for fresh view
                            fetch_gdp_sensitivity.clear()
                            fetch_year_trend.clear()

                    if not reply:
                        reply = "⚠️ No response generated. Please check your query format."
                except Exception as e:
                    reply = f"⚠️ Error: {str(e)}"
            st.markdown(reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    # ── Clear button ──────────────────────────────────────
    if msgs:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_messages = []
            st.rerun()


# =========================================================
# INTELLIGENCE PAGE
# =========================================================
def intelligence_page():
    p = st.session_state.last_prediction
    if p:
        display_val = format_prediction(p["value"], p["model"])
        st.markdown(f"""
        <div class="context-banner"><div class="subtle-note">
            <strong>Active Context</strong> &nbsp;·&nbsp;
            {html.escape(p['model'])} &nbsp;·&nbsp;
            {html.escape(p['country'])} &nbsp;·&nbsp;
            {p['year']} &nbsp;·&nbsp;
            {p['expenditure']}% GDP &nbsp;·&nbsp;
            {display_val}
        </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="context-banner" style="border-color:rgba(255,255,255,0.08);background:rgba(255,255,255,0.02);">
            <div class="subtle-note">No active prediction yet — or ask directly in the chat below.</div>
        </div>""", unsafe_allow_html=True)

    # ── Side-by-Side Layout ──────────────────────────────
    chart_col, chat_col = st.columns([1.1, 0.9], gap="medium")

    with chart_col:
        st.markdown('<div class="micro-label">Forecast Visualization</div>', unsafe_allow_html=True)
        if p:
            mode = st.radio("Chart view", ["GDP Sensitivity"],
                            horizontal=True, label_visibility="collapsed",
                            index=0, key="chart_mode_radio")
            render_forecast_chart(p, "gdp")
        else:
            render_empty_chart()

    with chat_col:
        render_chat_panel()


# =========================================================
# MAIN
# =========================================================
inject_css()
if st.session_state.logged_in:
    render_app_header()
    if st.session_state.page == "Forecast":
        forecast_page()
    else:
        intelligence_page()
else:
   login_page()
   