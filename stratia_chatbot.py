import os
import re
import requests
import spacy
from dotenv import load_dotenv

# =========================================================
# ENV + CONFIG
# =========================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LSTM_API     = os.getenv("LSTM_API",    "http://127.0.0.1:5001/predict")
TABULAR_API  = os.getenv("TABULAR_API", "http://127.0.0.1:5000/predict")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

COUNTRIES  = ["USA", "CHN", "GBR", "FRA", "RUS"]
INDICATORS = [
    "gdp", "inflation", "unemployment", "population",
    "health_expenditure", "education_expenditure",
    "co2_emissions", "asylum_seekers", "refugees",
]

nlp = spacy.load("en_core_web_sm")


# =========================================================
# SESSION STATE
# =========================================================
# `pending` holds partial info while STRATIA clarifies step-by-step.
# `last`    holds the last completed prediction for follow-up questions.
_session = {
    "pending": {
        "country":   None,
        "year":      None,
        "defense":   None,
        "indicator": None,
    },
    "awaiting": None,     # which field are we currently asking for?
    "last": {             # last completed prediction
        "country":    None,
        "year":       None,
        "defense":    None,
        "indicator":  None,
        "prediction": None,
    },
}

def _clear_pending():
    _session["pending"] = {k: None for k in _session["pending"]}
    _session["awaiting"] = None

def _save_last(country, year, defense, indicator, prediction):
    _session["last"] = {
        "country":    country,
        "year":       year,
        "defense":    defense,
        "indicator":  indicator,
        "prediction": prediction,
    }

def _has_last():
    return _session["last"]["prediction"] is not None

def _is_clarifying():
    return _session["awaiting"] is not None

def get_chat_last_prediction():
    return _session["last"]


# =========================================================
# INTENT DETECTION
# =========================================================
SECTION_KEYWORDS = {
    "explanation":    ["explain", "what does", "what is this", "meaning", "clarify", "elaborate"],
    "recommendation": ["recommend", "recommendation", "suggest", "advice", "what should", "what to do", "tip", "action"],
    "reasoning":      ["why", "reason", "cause", "because", "economic reasoning", "why does", "why would"],
    "impact":         ["impact", "effect", "consequence", "real world", "what happens", "outcome"],
    "summary":        ["summarize", "summary", "brief", "short", "quick", "overview", "tldr"],
}

def detect_sections(text):
    text_lower = text.lower()
    found = [s for s, kws in SECTION_KEYWORDS.items() if any(k in text_lower for k in kws)]
    return found if found else ["full"]

def is_general_question(text):
    text_lower = text.lower()
    triggers = [
        "what is", "how does", "define", "tell me about",
        "explain the concept", "difference between", "compare",
        "how do central banks", "what causes", "how is",
        "what are the effects of",
    ]
    has_trigger = any(t in text_lower for t in triggers)
    has_predict = any(k in text_lower for k in ["predict", "forecast", "will", "expected", "projection"])
    return has_trigger and not has_predict


# =========================================================
# NLP PARSER  — understands natural language aliases
# =========================================================

# Country aliases: maps any natural name / demonym → ISO code
COUNTRY_ALIASES = [
    (r"\busa\b",            "USA"), (r"\bunited states\b", "USA"),
    (r"\bamerica\b",        "USA"), (r"\bamerican\b",      "USA"),
    (r"\bchn\b",            "CHN"), (r"\bchina\b",         "CHN"),
    (r"\bchinese\b",        "CHN"),
    (r"\bgbr\b",            "GBR"), (r"\buk\b",            "GBR"),
    (r"\bunited kingdom\b", "GBR"), (r"\bbritain\b",       "GBR"),
    (r"\bgreat britain\b",  "GBR"), (r"\bengland\b",       "GBR"),
    (r"\bfra\b",            "FRA"), (r"\bfrance\b",        "FRA"),
    (r"\bfrench\b",         "FRA"),
    (r"\brus\b",            "RUS"), (r"\brussia\b",        "RUS"),
    (r"\brussian\b",        "RUS"),
]

# Indicator aliases: ordered list of (key, [phrases])
INDICATOR_ALIASES = [
    ("gdp",                  ["gdp", "gross domestic product", "economic growth",
                               "economic output", "growth rate"]),
    ("inflation",            ["inflation", "cpi", "consumer price index",
                               "price increase", "price level", "rising prices"]),
    ("unemployment",         ["unemployment", "jobless", "unemployment rate",
                               "jobless rate", "people out of work", "labor market"]),
    ("population",           ["population", "population size", "demographic",
                               "number of people", "people count"]),
    ("health_expenditure",   ["health expenditure", "health spending",
                               "healthcare expenditure", "healthcare spending",
                               "medical spending", "healthcare budget"]),
    ("education_expenditure",["education expenditure", "education spending",
                               "education budget", "school spending"]),
    ("asylum_seekers",       ["asylum seeker", "asylum seekers", "asylum applications",
                               "asylum"]),
    ("co2_emissions",        ["co2 emission", "co2 emissions", "co2",
                               "carbon emission", "carbon emissions", "carbon dioxide",
                               "greenhouse gas", "ghg", "carbon output",
                               "emission", "emissions"]),
    ("refugees",             ["refugee", "refugees", "displaced people",
                               "displaced persons", "forced migration"]),
]

# Phrases that indicate defense / military spending follows
DEFENSE_KEYWORDS = [
    "defense spending", "defence spending", "military spending",
    "military expenditure", "defense expenditure", "defence expenditure",
    "military budget", "defense budget", "defence budget",
    "armed forces spending", "defense outlay", "military outlay",
]

def parse_query(text):
    text_lower = text.lower()

    # ── Country ──────────────────────────────────────────────────────────
    country = None
    for pattern, code in COUNTRY_ALIASES:
        if re.search(pattern, text_lower):
            country = code
            break

    # ── Year ─────────────────────────────────────────────────────────────
    year_match = re.findall(r"\b(20\d{2})\b", text)
    year = int(year_match[0]) if year_match else None

    # ── Defense / military spending ───────────────────────────────────────
    defense = None
    # Try: any number followed by % or "percent"
    pct_match = re.findall(r"\b(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", text_lower)
    if pct_match:
        defense = float(pct_match[0])
    # Fallback: keyword followed by a number on the same line
    if defense is None:
        for kw in DEFENSE_KEYWORDS:
            if kw in text_lower:
                tail = text_lower.split(kw, 1)[1][:40]
                num  = re.findall(r"(\d+\.?\d*)", tail)
                if num:
                    defense = float(num[0])
                    break

    # ── Indicator ─────────────────────────────────────────────────────────
    indicator = None
    for key, aliases in INDICATOR_ALIASES:
        for alias in aliases:
            if alias in text_lower:
                indicator = key
                break
        if indicator:
            break

    return country, year, defense, indicator


# =========================================================
# CLARIFICATION FLOW
# =========================================================
FIELD_QUESTIONS = {
    "country": (
        "🌍 Which country are you interested in?\n"
        "   Supported: USA · CHN (China) · GBR (UK) · FRA (France) · RUS (Russia)"
    ),
    "year":      "📅 Which year would you like the prediction for? (e.g. 2028)",
    "indicator": (
        "📊 Which economic indicator do you want to predict?\n"
        "   Options: GDP · Inflation · Unemployment · Population · "
        "Health Expenditure · Education Expenditure · CO2 Emissions · "
        "Asylum Seekers · Refugees"
    ),
    "defense": (
        "🛡️ What defense spending (% of GDP) should I use for the model?\n"
        "   Type a number like 2.5 — or type 'skip' to use the default (2.0%)"
    ),
}

FIELD_ORDER = ["country", "year", "indicator", "defense"]

def _parse_field_answer(field, text):
    """Extract a single field value from the user's clarification reply.
    Reuses parse_query so aliases work identically in both flows."""
    text_stripped = text.strip()

    if field == "country":
        # reuse full parser so "china", "france" etc. all work
        country, _, _, _ = parse_query(text_stripped)
        return country

    if field == "year":
        m = re.findall(r"\b(20\d{2})\b", text_stripped)
        return int(m[0]) if m else None

    if field == "indicator":
        _, _, _, indicator = parse_query(text_stripped)
        return indicator

    if field == "defense":
        tl = text_stripped.lower()
        if "skip" in tl or "default" in tl or "no" == tl:
            return 2.0
        # accept "3.4", "3.4%", "3.4 percent"
        m = re.findall(r"(\d+\.?\d*)", text_stripped)
        return float(m[0]) if m else None

    return None

def _next_missing_field():
    pending = _session["pending"]
    for field in FIELD_ORDER:
        if pending[field] is None:
            return field
    return None

def _build_progress_line(p):
    parts = []
    if p["country"]:   parts.append(f"🌍 {p['country']}")
    if p["year"]:      parts.append(f"📅 {p['year']}")
    if p["indicator"]: parts.append(f"📊 {p['indicator'].upper()}")
    if p["defense"]:   parts.append(f"🛡️ {p['defense']}%")
    return ("Got it! So far: " + " · ".join(parts)) if parts else ""

def handle_clarification_reply(user_input):
    awaiting = _session["awaiting"]
    value    = _parse_field_answer(awaiting, user_input)

    if value is None:
        hint_map = {
            "country":   "Please type one of: USA, CHN, GBR, FRA, RUS",
            "year":      "Please enter a 4-digit year like 2028.",
            "indicator": "Try typing: gdp, inflation, unemployment, population, co2 emissions, refugees…",
            "defense":   "Type a number like 2.5, or type 'skip' to use the default.",
        }
        return (
            f"⚠️ I couldn't understand that. {hint_map[awaiting]}\n\n"
            f"{FIELD_QUESTIONS[awaiting]}"
        )

    _session["pending"][awaiting] = value

    next_field = _next_missing_field()
    if next_field:
        _session["awaiting"] = next_field
        progress = _build_progress_line(_session["pending"])
        return f"{progress}\n\n{FIELD_QUESTIONS[next_field]}"

    _session["awaiting"] = None
    p = _session["pending"]
    return _run_and_respond(p["country"], p["year"], p["indicator"], p["defense"])


# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def predict_gdp(country, year, defense):
    r = requests.post(LSTM_API, json={"model": "GDP Growth Forecast - LSTM", "expenditure": defense})
    return r.json()["prediction"]

def predict_inflation(country, year, defense):
    r = requests.post(LSTM_API, json={"model": "Inflation - LSTM", "expenditure": defense})
    return r.json()["prediction"]

def predict_unemployment(country, year, defense):
    r = requests.post(LSTM_API, json={"model": "Unemployment - LSTM", "expenditure": defense})
    return r.json()["prediction"]

def predict_health_expenditure(country, year, defense):
    r = requests.post(LSTM_API, json={"model": "Health Forecast - LSTM", "expenditure": defense})
    return r.json()["prediction"]

def predict_education_expenditure(country, year, defense):
    r = requests.post(LSTM_API, json={"model": "Education Forecast - LSTM", "expenditure": defense})
    return r.json()["prediction"]

def predict_population(country, year, defense):
    r = requests.post(LSTM_API, json={
        "model":   "Population Forecast - ARIMAX",
        "country": country,
        "year":    year,
        "gdp":     defense,
    })
    return r.json()["prediction"]

def predict_co2_emissions(country, year, defense):
    r = requests.post(TABULAR_API, json={
        "model":       "CO2 - Random Forest",
        "country":     country,
        "year":        year,
        "expenditure": defense,
    })
    return r.json()["prediction"]

def predict_asylum_seekers(country, year, defense):
    r = requests.post(TABULAR_API, json={
        "model":       "Asylum seekers - Xgboost",
        "country":     country,
        "year":        year,
        "expenditure": defense,
    })
    return r.json()["prediction"]

def predict_refugees(country, year, defense):
    r = requests.post(TABULAR_API, json={
        "model":       "Refugee - PKL",
        "country":     country,
        "year":        year,
        "expenditure": defense,
    })
    return r.json()["prediction"]

PREDICT_MAP = {
    "gdp":                   predict_gdp,
    "inflation":             predict_inflation,
    "unemployment":          predict_unemployment,
    "health_expenditure":    predict_health_expenditure,
    "education_expenditure": predict_education_expenditure,
    "population":            predict_population,
    "co2_emissions":         predict_co2_emissions,
    "asylum_seekers":        predict_asylum_seekers,
    "refugees":              predict_refugees,
}


# =========================================================
# EXPLANATION & RECOMMENDATION
# =========================================================
def generate_explanation(indicator, value):
    return {
        "gdp":                   "GDP growth reflects overall economic performance.",
        "inflation":             "Inflation represents the rate of increase in prices.",
        "unemployment":          "Unemployment reflects labor market conditions.",
        "population":            "Population trends influence economic demand and workforce size.",
        "health_expenditure":    "Health expenditure indicates investment in public health systems.",
        "education_expenditure": "Education expenditure reflects investment in human capital.",
        "co2_emissions":         "CO2 emissions indicate environmental impact and industrial activity.",
        "asylum_seekers":        "Asylum seekers reflect geopolitical and humanitarian conditions.",
        "refugees":              "Refugee numbers indicate displacement due to conflict or disaster.",
    }.get(indicator, "General economic indicator.")

def generate_recommendation(indicator, value, defense):
    rules = {
        "gdp": [
            (lambda v: v < 1,   "Boost economic growth through investment and fiscal stimulus."),
            (lambda v: v > 4,   "Monitor for overheating economy and inflation risks."),
            (lambda v: True,    "Maintain balanced economic policies."),
        ],
        "inflation": [
            (lambda v: v > 6,   "Consider tightening fiscal and monetary policies."),
            (lambda v: v < 2,   "Stimulate demand to avoid deflation."),
            (lambda v: True,    "Inflation is within a stable range."),
        ],
        "unemployment": [
            (lambda v: v > 7,   "Encourage job creation and investment in workforce development."),
            (lambda v: True,    "Labor market conditions are relatively stable."),
        ],
        "health_expenditure": [
            (lambda v: v < 3,   "Increase investment in healthcare infrastructure and services."),
            (lambda v: True,    "Health expenditure is adequate for current needs."),
        ],
        "education_expenditure": [
            (lambda v: v < 4,   "Increase funding for education to improve long-term growth."),
            (lambda v: True,    "Education investment is sufficient."),
        ],
        "co2_emissions": [
            (lambda v: v > 10,  "Implement stronger environmental regulations and promote clean energy."),
            (lambda v: True,    "CO2 emissions are relatively low."),
        ],
        "asylum_seekers": [
            (lambda v: v > 100000, "Enhance support for asylum seekers and address root causes of migration."),
            (lambda v: True,       "Asylum seeker numbers are manageable."),
        ],
        "population": [
            (lambda v: v > 100000000, "Plan for infrastructure and social services to accommodate population growth."),
            (lambda v: True,          "Population size is manageable."),
        ],
        "refugees": [
            (lambda v: v > 500000, "Increase humanitarian aid and support for refugee integration."),
            (lambda v: True,       "Refugee numbers are relatively low."),
        ],
    }
    for condition, message in rules.get(indicator, [(lambda v: True, "No specific recommendation available.")]):
        if condition(value):
            return message


# =========================================================
# GROQ LLM
# =========================================================
def _llm_call(system_prompt, user_prompt, max_tokens=400):
    try:
        payload = {
            "model":       "llama-3.3-70b-versatile",
            "temperature": 0.7,
            "max_tokens":  max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }
        response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=payload)
        result   = response.json()
        if "choices" not in result:
            return f"AI insight unavailable: {result.get('error', {}).get('message', str(result))}"
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"AI insight temporarily unavailable. ({str(e)})"

SYSTEM_PROMPT = (
    "You are STRATIA, an advanced AI economic advisor. "
    "Be concise, professional, and insightful."
)

def llm_full_insight(country, indicator, value, defense, year):
    prompt = f"""
Analyze the following prediction:

Country: {country} | Year: {year} | Indicator: {indicator}
Predicted Value: {value:.2f} | Defense Spending: {defense}%

Provide:
1. Clear explanation of what this means
2. Economic reasoning (why this might occur)
3. Real-world impact
4. Practical recommendation

Keep it concise and professional.
"""
    return _llm_call(SYSTEM_PROMPT, prompt)

def llm_section(section, country, indicator, value, defense, year):
    ctx = (
        f"Country: {country}, Year: {year}, Indicator: {indicator}, "
        f"Predicted Value: {value:.2f}, Defense Spending: {defense}%"
    )
    prompts = {
        "explanation":    f"Given — {ctx} — explain in 2-3 sentences what this predicted value means in plain terms.",
        "recommendation": f"Given — {ctx} — provide only the practical policy and action recommendations. Be specific.",
        "reasoning":      f"Given — {ctx} — explain only the economic reasoning behind why this outcome might occur.",
        "impact":         f"Given — {ctx} — describe only the real-world impact on citizens, businesses, and government.",
        "summary":        f"Given — {ctx} — give a 2-sentence executive summary and the single most important takeaway.",
    }
    return _llm_call(SYSTEM_PROMPT, prompts.get(section, f"Comment briefly on: {ctx}"), max_tokens=250)

def llm_general_question(question):
    prompt = (
        f"The user asked: \"{question}\"\n\n"
        "Answer clearly and concisely in 3-5 sentences. "
        "Do not reference any specific model prediction."
    )
    return _llm_call(SYSTEM_PROMPT, prompt, max_tokens=300)


# =========================================================
# RESPONSE BUILDERS
# =========================================================
def build_full_response(country, year, indicator, prediction, defense):
    explanation    = generate_explanation(indicator, prediction)
    recommendation = generate_recommendation(indicator, prediction, defense)
    ai_text        = llm_full_insight(country, indicator, prediction, defense, year)
    return (
        f"📊 STRATIA Economic Insight\n\n"
        f"🌍 Country: {country}  |  📅 Year: {year}\n"
        f"🔹 Indicator: {indicator.upper()}  |  📈 Predicted Value: {prediction:.2f}\n\n"
        f"📖 Explanation:\n{explanation}\n\n"
        f"💡 Recommendation:\n{recommendation}\n\n"
        f"🧠 AI Enhanced Insight:\n{ai_text}\n"
    )

def build_section_response(sections, country, year, indicator, prediction, defense):
    header = (
        f"📊 STRATIA — {indicator.upper()} for {country} ({year})\n"
        f"📈 Predicted Value: {prediction:.2f}\n"
    )
    parts = [header]
    for section in sections:
        if section == "explanation":
            rb = generate_explanation(indicator, prediction)
            ai = llm_section("explanation", country, indicator, prediction, defense, year)
            parts.append(f"📖 Explanation:\n{rb}\n\n🧠 Detail:\n{ai}")
        elif section == "recommendation":
            rb = generate_recommendation(indicator, prediction, defense)
            ai = llm_section("recommendation", country, indicator, prediction, defense, year)
            parts.append(f"💡 Recommendation:\n{rb}\n\n🧠 AI Recommendations:\n{ai}")
        elif section == "reasoning":
            ai = llm_section("reasoning", country, indicator, prediction, defense, year)
            parts.append(f"🔍 Economic Reasoning:\n{ai}")
        elif section == "impact":
            ai = llm_section("impact", country, indicator, prediction, defense, year)
            parts.append(f"🌐 Real-World Impact:\n{ai}")
        elif section == "summary":
            ai = llm_section("summary", country, indicator, prediction, defense, year)
            parts.append(f"📝 Summary:\n{ai}")
    return "\n\n".join(parts)


# =========================================================
# CORE PREDICTION RUNNER
# =========================================================
def _run_and_respond(country, year, indicator, defense):
    _clear_pending()
    try:
        predict_fn = PREDICT_MAP.get(indicator)
        if not predict_fn:
            return "❌ Indicator not supported."
        prediction = predict_fn(country, year, defense)
    except requests.exceptions.ConnectionError:
        return (
            "⚠️ Cannot connect to prediction API servers. "
            "Make sure api_lstm.py (port 5001) and api_tabular.py (port 5000) are running."
        )
    except Exception as e:
        return f"⚠️ Prediction error: {str(e)}"

    _save_last(country, year, defense, indicator, prediction)
    return build_full_response(country, year, indicator, prediction, defense)


# =========================================================
# MAIN CHATBOT FUNCTION  (called by Streamlit / CLI)
# =========================================================
def chatbot_response(user_input):
    user_input = user_input.strip()

    # ── 0. Mid-clarification reply ────────────────────────────────────────
    # Only treat as a clarification reply if we're genuinely mid-flow AND
    # the new message doesn't look like a complete fresh query.
    if _is_clarifying():
        country, year, defense, indicator = parse_query(user_input)
        # If the user typed a full new query, abandon clarification and run it
        if country and year and indicator:
            _clear_pending()
            if defense is None:
                defense = 2.0
            return _run_and_respond(country, year, indicator, defense)
        return handle_clarification_reply(user_input)

    # ── 1. General economic question ──────────────────────────────────────
    if is_general_question(user_input):
        return f"🧠 STRATIA:\n\n{llm_general_question(user_input)}"

    # ── 2. Parse the query ────────────────────────────────────────────────
    country, year, defense, indicator = parse_query(user_input)

    # ── 3. Follow-up on last prediction ──────────────────────────────────
    if not country and not year and not indicator and _has_last():
        last     = _session["last"]
        sections = detect_sections(user_input)
        if sections == ["full"]:
            return build_full_response(
                last["country"], last["year"], last["indicator"],
                last["prediction"], last["defense"],
            )
        return build_section_response(
            sections,
            last["country"], last["year"], last["indicator"],
            last["prediction"], last["defense"],
        )

    # ── 4. All required info present → run immediately ────────────────────
    if country and year and indicator:
        if defense is None:
            defense = 2.0
        return _run_and_respond(country, year, indicator, defense)

    # ── 5. Some info missing → start step-by-step clarification ──────────
    _session["pending"]["country"]   = country
    _session["pending"]["year"]      = year
    _session["pending"]["indicator"] = indicator
    _session["pending"]["defense"]   = defense

    first_missing = _next_missing_field()
    _session["awaiting"] = first_missing

    detected = []
    if country:   detected.append(f"🌍 {country}")
    if year:      detected.append(f"📅 {year}")
    if indicator: detected.append(f"📊 {indicator.upper()}")

    prefix = (
        ("Got it! I have: " + " · ".join(detected) + "\n\nNow, ")
        if detected else
        "I'd love to help with a prediction! I'll need a few details.\n\n"
    )
    return prefix + FIELD_QUESTIONS[first_missing]


# =========================================================
# CLI
# =========================================================
if __name__ == "__main__":
    print("\n🚀 STRATIA AI Assistant Ready!")
    print("Ask a prediction, follow-up, or a general economic question.")
    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("Ask STRATIA: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("👋 Exiting STRATIA...")
            break
        try:
            print("\n" + chatbot_response(user_input) + "\n")
        except Exception as e:
            print("⚠️ Error:", str(e))