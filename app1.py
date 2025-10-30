# app1_fast_semantic.py
"""
Streamlit app — Fast Local CSV NL Query with semantic search & caching
- Startup: load CSVs, build schema, synonyms and lightweight TF-IDF embeddings once
- Per-query: fast matching -> small prompt -> single LLM call
- Safety: sanitize model code + validate before exec, with local fallback engine
- Cache: persistent spec -> result cache to speed repeated queries

Minimal external packages required:
- streamlit
- pandas
- numpy
- ollama (client for local model)
- scikit-learn (for TfidfVectorizer & cosine similarity)

Drop this file into your project and run with `streamlit run app1_fast_semantic.py`.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
import json
import re
import hashlib
import time
import pickle
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List

import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------
# Config
# -------------------------
DATA_DIR = "data"
CACHE_FILE = "query_cache.pkl"  # persistent simple cache
RESULTS_DIR = "results"
MODEL_NAME = "phi3-mini-local"  # change to your preferred local model
OLLAMA_MODEL = MODEL_NAME

# Allowed runtime globals for exec
ALLOWED_GLOBALS = {"pd": pd, "np": np}

st.set_page_config(page_title="Fast Local CSV NL Query", layout="wide")

# -------------------------
# Synonyms (extend as needed)
# -------------------------
# -------------------------
# COLUMN VALUE SYNONYMS
# -------------------------
# ---------------------------------
# 1️⃣ COLUMN VALUE SYNONYMS
# ---------------------------------
farmer_synonyms = {
    "farmer_id": {
        "Farmer ID": ["id", "farmerid", "farmer code", "unique id", "farmer number"]
    },
    "farmer_name": {
        "Farmer Name": ["name", "farmername", "person name", "owner name", "farmer"]
    },
    "aadhaar": {
        "Aadhaar": ["aadhaar", "aadhar", "uid", "aadhar number", "identity number"]
    },
    "aadhaar_hash": {
        "Aadhaar Hash": ["aadhar hash", "aadhaarhash", "encrypted aadhar", "masked aadhar"]
    },
    "dob": {
        "Date of Birth": ["dob", "birthdate", "date of birth", "birthday", "birth year", "age"]
    },
    "gender": {
        "Female": ["female", "f", "woman", "women", "lady", "ladies"],
        "Male": ["male", "m", "man", "men", "gentleman", "gentlemen"],
        "Other": ["other", "third gender", "transgender", "non-binary"]
    },
    "state": {
        "Andhra Pradesh": ["andhra", "andhrapradesh", "ap"],
        "Arunachal Pradesh": ["arunachal", "arunachalpradesh"],
        "Assam": ["assam", "asm"],
        "Bihar": ["bihar", "bhr"],
        "Chhattisgarh": ["chhattisgarh", "cg"],
        "Goa": ["goa"],
        "Gujarat": ["gujarat", "gj"],
        "Haryana": ["haryana", "hr"],
        "Himachal Pradesh": ["himachal", "hp"],
        "Jharkhand": ["jharkhand", "jh"],
        "Karnataka": ["karnataka", "kar", "ka", "k’taka"],
        "Kerala": ["kerala", "kl"],
        "Madhya Pradesh": ["mp", "madhyapradesh", "madhya"],
        "Maharashtra": ["maharashtra", "mh", "maha"],
        "Manipur": ["manipur", "mn"],
        "Meghalaya": ["meghalaya", "ml"],
        "Mizoram": ["mizoram", "mz"],
        "Nagaland": ["nagaland", "nl"],
        "Odisha": ["odisha", "orissa", "od", "odr"],
        "Punjab": ["punjab", "pb"],
        "Rajasthan": ["rajasthan", "raj", "rj"],
        "Sikkim": ["sikkim", "sk"],
        "Tamil Nadu": ["tamilnadu", "tn", "madras"],
        "Telangana": ["telangana", "tg", "tel"],
        "Tripura": ["tripura", "tr"],
        "Uttar Pradesh": ["up", "uttarpradesh"],
        "Uttarakhand": ["uttarakhand", "uk", "ua"],
        "West Bengal": ["wb", "bengal", "westbengal"]
    },
    "district": {
        "District": ["district", "city", "zone", "region", "area"]
    },
    "village": {
        "Village": ["village", "village name", "locality", "hamlet", "rural area", "settlement"]
    },
    "mobile_no": {
        "Mobile Number": ["mobile", "phone", "phone number", "contact", "contact number"]
    },
    "address": {
        "Address": ["address", "location", "place", "residence", "home"]
    },
    "farmer_category": {
        "Marginal": ["marginal", "very small", "tiny", "smallest"],
        "Small": ["small", "smaller", "minor"],
        "Big": ["large", "big", "large land", "big land", "major", "large scale"]
    },
    "caste_category": {
        "SC": ["sc", "scheduled caste", "scheduled-caste"],
        "ST": ["st", "scheduled tribe", "scheduled-tribe"],
        "OBC": ["obc", "other backward class", "other-backward", "backward caste"],
        "General": ["general", "gen", "open", "upper caste"]
    },
    "farm_area_ha": {
        "Farm Area (ha)": ["land", "farm area", "area", "hectare", "ha", "acre", "land size", "farm size", "plot size"]
    },
    "crop": {
        "Paddy": ["paddy", "rice", "basmati", "non-basmati", "parboiled rice"],
        "Wheat": ["wheat", "durum", "atta wheat", "bread wheat"],
        "Maize": ["maize", "corn", "sweetcorn", "babycorn"],
        "Millets": ["millet", "bajra", "ragi", "jowar", "sorghum", "foxtail", "barnyard"],
        "Sugarcane": ["sugarcane", "cane"],
        "Cotton": ["cotton", "bt cotton", "kapas"],
        "Pulses": ["pulse", "pulses", "dal", "lentil", "gram", "tur", "moong", "masoor", "urad"],
        "Oilseeds": ["oilseed", "mustard", "groundnut", "soybean", "sunflower", "sesame"],
        "Fruits": ["fruit", "fruits", "banana", "mango", "orange", "grape", "guava", "apple"],
        "Vegetables": ["vegetable", "vegetables", "tomato", "onion", "potato", "brinjal", "chili", "cabbage", "cauliflower"],
        "Rice":["rice","Rice"]
    }
}

seed_synonyms = {
    "Crop Name": ["crop", "plant", "seed crop", "variety", "crop name"],
    "Scientific name": ["scientific name", "botanical name", "species", "latin name"],
    "Seed Code": ["seed code", "id", "seed id", "batch number", "code"],
    "soil_type": ["soil", "soil type", "land type", "terrain", "soil condition"],
    "irrigation_source": ["irrigation", "water source", "watering", "irrigation type", "source of irrigation"],
    "fertilizer_used": ["fertilizer", "manure", "chemical used", "fertilizer type", "nutrient added"],
    "seed_type": ["seed", "seed category", "type of seed", "hybrid/local", "certified/uncertified"],
    "weather_condition": ["weather", "climate", "rainfall", "season", "weather type"],
    "ownership_type": ["ownership", "land ownership", "property type", "farmer type"],
    "loan_taken": ["loan", "credit", "borrowed", "financing", "debt"],
    "insurance_status": ["insurance", "insured", "policy", "coverage", "insurance type"],
    "pesticide_usage": ["pesticide", "chemical spray", "pesticide use", "insecticide"],
    "last_sowing_date": ["sowing date", "planting date", "date of sowing", "last sowing"],
    "expected_harvest_date": ["harvest date", "expected harvest", "yield date", "crop ready date"],

    "soil_type_values": {
        "Loamy": ["loamy", "loam", "medium soil"],
        "Clay": ["clay", "clayey", "heavy soil"],
        "Sandy": ["sandy", "sand", "light soil"],
        "Black": ["black", "regur", "cotton soil"],
        "Red": ["red", "laterite"],
        "Alluvial": ["alluvial", "river soil"]
    },

    "irrigation_source_values": {
        "Canal": ["canal", "irrigation canal"],
        "Tube Well": ["tubewell", "borewell", "well", "pump"],
        "Rainfed": ["rainfed", "rainwater", "monsoon"],
        "Tank": ["tank", "pond", "reservoir"],
        "Drip": ["drip", "micro irrigation"]
    },

    "fertilizer_used_values": {
        "Urea": ["urea", "nitrogen fertilizer"],
        "DAP": ["dap", "di-ammonium phosphate"],
        "Compost": ["compost", "organic fertilizer", "farmyard manure", "cow dung"],
        "NPK": ["npk", "mixed fertilizer", "complex fertilizer"]
    },

    "seed_type_values": {
        "Hybrid": ["hybrid", "crossbred"],
        "Certified": ["certified", "approved", "govt certified"],
        "Foundation": ["foundation", "breeder"],
        "Local": ["local", "desi", "traditional"]
    },

    "weather_condition_values": {
        "Dry": ["dry", "arid", "low rainfall"],
        "Humid": ["humid", "moist", "wet"],
        "Normal": ["normal", "moderate climate", "average"],
        "Hot": ["hot", "warm"],
        "Cold": ["cold", "chilly", "frost"]
    },

    "ownership_type_values": {
        "Owned": ["owned", "self-owned"],
        "Leased": ["leased", "rented", "tenant"],
        "Joint": ["joint", "shared"]
    },

    "loan_taken_values": {
        "Yes": ["yes", "loan taken", "borrowed", "has loan"],
        "No": ["no", "no loan", "debt free"]
    },

    "insurance_status_values": {
        "Insured": ["insured", "covered", "policy active"],
        "Not Insured": ["not insured", "no insurance", "uninsured"]
    },

    "pesticide_usage_values": {
        "Yes": ["yes", "pesticide used", "chemicals applied"],
        "No": ["no", "organic", "no pesticide"]
    }
}

# ---------------- AUTO SELECT SYNONYMS ----------------
if dataset_name:
    if "seed" in dataset_name.lower():
        SYNONYMS = seed_synonyms
        st.info("🌱 Using seed dataset synonyms.")
    else:
        SYNONYMS = farmer_synonyms
        st.info("👩‍🌾 Using farmer dataset synonyms.")
else:
    SYNONYMS = farmer_synonyms



# ---------------------------------
# 2️⃣ INTENT SYNONYMS
# ---------------------------------
INTENT_SYNONYMS = {
    "list": ["list", "show", "display", "give", "get", "fetch", "retrieve", "see", "view", "return"],
    "count": ["count", "number", "how many", "total", "sum of", "quantity of", "no. of"],
    "filter": ["filter", "find", "search for", "look for", "select", "narrow down to"],
    "compare": ["compare", "versus", "difference between", "contrast", "diff"],
    "group": ["group by", "aggregate", "cluster", "summarize", "breakdown by"],
    "sort": ["sort", "order", "arrange", "rank", "top", "highest", "lowest"],
    "average": ["average", "mean", "avg", "on average", "typically"],
    "max": ["max", "maximum", "highest", "largest", "biggest"],
    "min": ["min", "minimum", "lowest", "smallest", "least"],
    "sum": ["sum", "total", "combined", "overall"]
}




# Flatten synonyms for lookup
FLAT_SYNONYMS: Dict[str, Tuple[str, str]] = {}
for col, mapping in SYNONYMS.items():
    for canonical, words in mapping.items():
        for w in words:
            FLAT_SYNONYMS[w.lower()] = (col, canonical)

# -------------------------
# Utilities: Disk cache
# -------------------------

def load_cache(path: str = CACHE_FILE) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def save_cache(cache: dict, path: str = CACHE_FILE):
    try:
        with open(path, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass

# in-memory cache for speed
CACHE = load_cache()

# -------------------------
# Startup: load datasets & build light embeddings
# -------------------------

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


@st.cache_data(show_spinner=False)
def load_all_csvs_from_dir(data_dir=DATA_DIR) -> Dict[str, pd.DataFrame]:
    datasets = {}
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return datasets
    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".csv"):
            path = os.path.join(data_dir, filename)
            try:
                df = pd.read_csv(path)
                name = os.path.splitext(filename)[0]
                datasets[name] = df
            except Exception:
                # skip broken files
                continue
    return datasets


@st.cache_resource(show_spinner=False)
def build_column_corpus_and_vectorizer(datasets: Dict[str, pd.DataFrame]):
    """
    Build a TF-IDF vectorizer over textual descriptions of each dataset column.
    Returns:
      - vectorizer
      - mapping: dataset -> list of column_texts
      - mapping: dataset -> list of column_names (ordered same as texts)
    """
    corpus_texts = []
    index_map = []  # tuples (dataset, column)

    for ds_name, df in datasets.items():
        for col in df.columns:
            # build a small text describing the column: name + up to 5 example values
            example_vals = []
            try:
                vals = df[col].dropna().unique()
                if len(vals) > 0:
                    # sample up to 5 representative values
                    sample_count = min(5, len(vals))
                    # prefer small unique sets to show categories
                    if len(vals) <= 50:
                        sample_vals = [str(v) for v in vals[:sample_count]]
                    else:
                        sample_vals = [str(v) for v in np.random.choice(vals, sample_count, replace=False)]
                    example_vals = sample_vals
            except Exception:
                example_vals = []
            text = f"{col} " + " ".join(example_vals)
            corpus_texts.append(text)
            index_map.append((ds_name, col, text))

    if not corpus_texts:
        # no data — return empty structures
        vectorizer = TfidfVectorizer().fit(["empty"])
        return vectorizer, {}, {}

    vectorizer = TfidfVectorizer().fit(corpus_texts)

    # Build per-dataset structures
    ds_col_texts: Dict[str, List[str]] = {}
    ds_col_names: Dict[str, List[str]] = {}
    for ds_name, col, text in index_map:
        ds_col_texts.setdefault(ds_name, []).append(text)
        ds_col_names.setdefault(ds_name, []).append(col)

    return vectorizer, ds_col_texts, ds_col_names


# -------------------------
# Lightweight NL normalization & local rewrite
# -------------------------

def normalize_user_query(user_query: str) -> str:
    """Replace known synonyms in the user query with canonical hints.
    e.g. "women" -> "gender==Female" (this helps the model).
    This function only performs conservative rewrites — not heavy NLP.
    """
    q = user_query.lower()
    # quick fixes
    q = q.replace("sc farmers", "sc")
    q = q.replace("st farmers", "st")

    # replace word tokens that match synonyms
    tokens = re.split(r"(\W+)", q)
    out = []
    for t in tokens:
        if not t.strip():
            out.append(t)
            continue
        key = t.strip().lower()
        if key in FLAT_SYNONYMS:
            col, canonical = FLAT_SYNONYMS[key]
            out.append(f"{col}=={canonical}")
        else:
            out.append(t)
    normalized = "".join(out)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized

def detect_intent(user_query: str):
    """
    Detects what kind of action the user wants (count/list/sort/etc.)
    based on INTENT_SYNONYMS.
    """
    query_lower = user_query.lower()
    for intent, keywords in INTENT_SYNONYMS.items():
        for word in keywords:
            if word in query_lower:
                return intent
    return "list"  # default intent if nothing matched

# -------------------------
# Matching pipeline: find top relevant columns for a query
# -------------------------

def find_top_columns_for_query(query: str, dataset_name: str, vectorizer: TfidfVectorizer,
                               ds_col_texts: Dict[str, List[str]], ds_col_names: Dict[str, List[str]], top_k=6):
    """
    Returns list of (column_name, score, example_text) for the dataset
    """
    if dataset_name not in ds_col_texts:
        return []
    texts = ds_col_texts[dataset_name]
    names = ds_col_names[dataset_name]

    q_vec = vectorizer.transform([query])
    texts_vec = vectorizer.transform(texts)
    sims = cosine_similarity(q_vec, texts_vec)[0]
    idx = np.argsort(-sims)[:top_k]
    results = []
    for i in idx:
        results.append((names[i], float(sims[i]), texts[i]))
    return results


# -------------------------
# Prompt builder: small, structured prompt
# -------------------------

def build_minimal_prompt(user_query: str, dataset_name: str, top_cols: List[Tuple[str, float, str]], intent: str = "list"):
    cols_str = ", ".join([f"'{c[0]}'" for c in top_cols]) if top_cols else ""
    examples = "\n".join([f"- {c[0]}: {c[2]}" for c in top_cols]) if top_cols else ""

    prompt = (
        "You are an expert Python pandas assistant. Respond with code only.\n"
        f"Operate on datasets['{dataset_name}'].\n"
        f"Use these likely-relevant columns: [{cols_str}]\n"
        "Some examples: \n"
        f"{examples}\n"
        f"The user wants to {intent} the data based on their request.\n"   # <-- ADD THIS LINE HERE
        "User request:\n"
        f"{user_query}\n\n"
        "Requirements:\n"
        "- Write only Python code; do not include any explanation.\n"
        f"- Start with: result = datasets['{dataset_name}'].copy()\n"
        "- Use result = result.query(...) or pandas indexing to filter.\n"
        "- Assign final DataFrame (or scalar) to `result`.\n"
        "Code:\n"
    )
    return prompt



# -------------------------
# Ollama call
# -------------------------

def call_ollama(prompt: str, timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    try:
        res = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful Python pandas assistant. Respond with code only."},
                {"role": "user", "content": prompt}
            ],
        )
        text = ""
        try:
            text = res.get("message", {}).get("content", "")
        except Exception:
            text = str(res)
        return text, None
    except Exception as e:
        return None, str(e)


# -------------------------
# Model output sanitizers & safety checks
# -------------------------

FORBIDDEN_PATTERNS = [
    r"\bimport\b",
    r"\bos\b",
    r"\bsubprocess\b",
    r"\beval\b",
    r"\bexec\b",
    r"__",  # dunder
    r"open\(",
    r"requests\.",
    r"socket\.",
    r"pickle\.",
    r"shutil",
]


def is_code_safe(code_text: str) -> Tuple[bool, Optional[str]]:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code_text):
            return False, f"Forbidden token or pattern found: {pat}"
    # require assignment to `result`
    if "result" not in code_text:
        return False, "No 'result' variable detected in generated code."
    return True, None


def sanitize_code_text(code_text: str) -> str:
    if not isinstance(code_text, str):
        code_text = str(code_text)
    code_text = code_text.replace("’", "'").replace("‘", "'")
    code_text = code_text.replace("“", '"').replace("”", '"')
    code_text = code_text.encode("ascii", "ignore").decode()
    # remove fenced triples
    code_text = re.sub(r"^```(?:python)?\s*", "", code_text, flags=re.I)
    code_text = re.sub(r"\s*```$", "", code_text)
    return code_text.strip()


def extract_code_from_model(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"```python(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: return entire text
    return text.strip()


def safe_exec_user_code(code_text: str, datasets: dict, dataset_name: str):
    """
    Execute code in restricted environment and return (result_obj, used_code, error_str).
    """
    if not code_text:
        return None, code_text, "No code provided"
    code_text = sanitize_code_text(code_text)

    safe, reason = is_code_safe(code_text)
    if not safe:
        return None, code_text, f"Code rejected by safety checks: {reason}"

    # Provide only safe globals + datasets
    allowed_globals = dict(ALLOWED_GLOBALS)
    allowed_globals["datasets"] = datasets

    # prepopulate result with a safe copy
    if dataset_name not in datasets:
        return None, code_text, f"Dataset '{dataset_name}' not found"
    local_env = {"result": datasets[dataset_name].copy()}

    try:
        exec(code_text, allowed_globals, local_env)
        return local_env.get("result", None), code_text, None
    except Exception as e:
        return None, code_text, str(e)


# -------------------------
# Simple local fallback engine (basic filters only)
# -------------------------

def apply_local_simple_filters(user_query: str, datasets: dict, dataset_name: str):
    """
    Attempt to parse simple equality and numeric comparisons (very conservative) from the normalized query.
    It looks for patterns like "col==Value" or "col==Value AND col2>2" which our normalize_user_query may produce.
    Returns (result_df or scalar, reason_string)
    """
    normalized = normalize_user_query(user_query)
    # find simple binary operations: column==Value or column==Value
    # we'll split by ' and ' or ' & '
    tokens = re.split(r"\band\b|&|\\n", normalized)
    if not tokens:
        return None, "No simple filters found"

    if dataset_name not in datasets:
        return None, f"Dataset '{dataset_name}' missing"

    df = datasets[dataset_name].copy()
    applied_any = False

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        # equality
        m_eq = re.match(r"^([A-Za-z0-9_]+)\s*==\s*([A-Za-z0-9_\- ]+)$", tok)
        if m_eq:
            col, val = m_eq.group(1), m_eq.group(2).strip()
            # try to apply
            if col in df.columns:
                # try numeric
                try:
                    num = float(val)
                    df = df[df[col].astype(float) == num]
                except Exception:
                    # string compare (case-insensitive)
                    df = df[df[col].astype(str).str.lower() == val.lower()]
                applied_any = True
                continue
        # numeric comparators: >, >=, <, <=
        m_num = re.match(r"^([A-Za-z0-9_]+)\s*(>=|<=|>|<)\s*([0-9\.]+)$", tok)
        if m_num:
            col, op, val = m_num.group(1), m_num.group(2), float(m_num.group(3))
            if col in df.columns:
                try:
                    if op == ">":
                        df = df[df[col].astype(float) > val]
                    elif op == ">=":
                        df = df[df[col].astype(float) >= val]
                    elif op == "<":
                        df = df[df[col].astype(float) < val]
                    elif op == "<=":
                        df = df[df[col].astype(float) <= val]
                    applied_any = True
                    continue
                except Exception:
                    continue
    if not applied_any:
        return None, "No conservative filter could be applied locally"
    return df, "applied local filters"


# -------------------------
# Standardize output filename
# -------------------------

def save_result_standard(result_obj: Any) -> Optional[str]:
    try:
        ensure_dirs()
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        filename = f"result_{ts}.csv"
        path = os.path.join(RESULTS_DIR, filename)
        if isinstance(result_obj, pd.DataFrame):
            result_obj.to_csv(path, index=False)
            return path
        else:
            # for scalar or other objects, save repr
            with open(path, "w", encoding="utf-8") as f:
                f.write(repr(result_obj))
            return path
    except Exception:
        return None


# -------------------------
# UI & orchestration
# -------------------------

ensure_dirs()

st.title("Fast Local CSV NL Query — Ollama + pandas (semantic)")
st.write("Startup: building compact schema + lightweight embeddings. Per-query: fast matching + short prompt.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data")
    ensure_dirs()
    uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
    if uploaded_files:
        for f in uploaded_files:
            path = os.path.join(DATA_DIR, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(uploaded_files)} files to {DATA_DIR}")

    datasets = load_all_csvs_from_dir()

        # --- Remove uploaded CSV files ---
    st.subheader("Remove uploaded CSV")
    existing_csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if existing_csvs:
        file_to_delete = st.selectbox("Select a file to delete", existing_csvs, index=0)
        if st.button("Delete selected file"):
            try:
                os.remove(os.path.join(DATA_DIR, file_to_delete))
                st.success(f"Deleted {file_to_delete}")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")
    else:
        st.info("No CSV files found to delete.")

    if not datasets:
        st.info("No CSVs found in data/. For quick testing, drop a CSV or use a default empty schema.")
        datasets = {"farmer_data_dummy": pd.DataFrame(columns=[
            "farmer_id","farmer_name","aadhaar","aadhaar_hash","dob","gender",
            "state_lgd_code","district_lgd_code","sub_district_lgd_code",
            "village_lgd_code","mobile_no","address","farmer_category",
            "caste_category","farm_area_ha","crop"
        ])}

    st.subheader("Loaded datasets")
    for name, df in datasets.items():
        with st.expander(f"{name} — {len(df)} rows, {len(df.columns)} cols"):
            info_df = pd.DataFrame({"column": df.columns, "dtype": [str(dt) for dt in df.dtypes]})
            st.dataframe(info_df)

    st.markdown("---")
    st.checkbox("Show generated python code before execution", key="show_code", value=True)
    st.checkbox("Allow merges/joins (model may generate merges if asked)", key="allow_merges", value=False)
    st.write(f"Using model: `{OLLAMA_MODEL}` (set MODEL_NAME variable in script to change)")

# Build vectorizer + per-dataset column corpus
vectorizer, ds_col_texts, ds_col_names = build_column_corpus_and_vectorizer(datasets)

with col2:
    st.header("Ask a question")
    user_query = st.text_area("Natural language query (e.g. 'list sc women with >2 ha')", height=120)
    run = st.button("Run query")

    if run:
        if not user_query:
            st.error("Enter a query")
        else:
            # select dataset
            dataset_name = "farmer_data_dummy" if "farmer_data_dummy" in datasets else list(datasets.keys())[0]

            # 1) Normalize user query (lightweight)
            normalized = normalize_user_query(user_query)

            # 2) Fast semantic match to discover top columns
            top_cols = find_top_columns_for_query(normalized, dataset_name, vectorizer, ds_col_texts, ds_col_names, top_k=6)

            # 3) Build prompt
            prompt = build_minimal_prompt(normalized, dataset_name, top_cols)

            st.subheader("Prompt (sent to model)")
            with st.expander("View prompt"):
                st.code(prompt)

            # 4) Compute cache key (spec cache)
            key_obj = {
                "dataset": dataset_name,
                "normalized_query": normalized,
                "top_cols": [c[0] for c in top_cols]
            }
            key_raw = json.dumps(key_obj, sort_keys=True)
            key_hash = hashlib.sha256(key_raw.encode()).hexdigest()

            # 5) Check cache
            cached = CACHE.get(key_hash)
            if cached is not None:
                st.success("Cache hit — returning cached result")
                try:
                    # cached stored as CSV bytes
                    df_cached = pd.read_csv(pd.io.common.BytesIO(cached))
                    st.dataframe(df_cached)
                    st.download_button("Download CSV", data=cached, file_name="cached_result.csv")
                except Exception:
                    st.write(cached)
                st.stop()

            # 6) Call LLM
            with st.spinner("Calling model..."):
                model_text, err = call_ollama(prompt)

            if err:
                st.error(f"Model call failed: {err}")
                # try local fallback
                fallback_df, reason = apply_local_simple_filters(user_query, datasets, dataset_name)
                if fallback_df is not None:
                    st.info("Local fallback succeeded — returning local result")
                    st.dataframe(fallback_df)
                    csv_bytes = fallback_df.to_csv(index=False).encode("utf-8")
                    CACHE[key_hash] = csv_bytes
                    save_cache(CACHE)
                    st.download_button("Download CSV", data=csv_bytes, file_name="result_fallback.csv")
                else:
                    st.error("Local fallback could not produce a result.")
                st.stop()

            # 7) Extract and sanitize code
            code_candidate = extract_code_from_model(model_text)
            code_candidate = sanitize_code_text(code_candidate)

            if st.session_state.get("show_code"):
                st.subheader("Generated code (sanitized)")
                st.code(code_candidate)

            # 8) Try executing safely
            result_obj, used_code, exec_err = safe_exec_user_code(code_candidate, datasets, dataset_name)

            if exec_err:
                st.error(f"Error executing generated code: {exec_err}")
                # show raw model output to help debugging
                st.markdown("Model raw text:")
                st.text_area("Model raw text", value=model_text, height=220)
                # Attempt local fallback
                fallback_df, reason = apply_local_simple_filters(user_query, datasets, dataset_name)
                if fallback_df is not None:
                    st.info("Local fallback succeeded — returning local result")
                    st.dataframe(fallback_df)
                    csv_bytes = fallback_df.to_csv(index=False).encode("utf-8")
                    CACHE[key_hash] = csv_bytes
                    save_cache(CACHE)
                    st.download_button("Download CSV", data=csv_bytes, file_name="result_fallback.csv")
                else:
                    st.error("Local fallback could not produce a result.")
                st.stop()

            # 9) Success — show result and cache
            if result_obj is None:
                st.warning("Model code produced no result. Showing raw model output instead.")
                st.text_area("Model raw text", value=model_text, height=220)
            else:
                st.subheader("Result — `result`")
                try:
                    if isinstance(result_obj, pd.DataFrame):
                        st.dataframe(result_obj)
                        csv_bytes = result_obj.to_csv(index=False).encode("utf-8")

                        # standardize filename and save to results/
                        saved_path = save_result_standard(result_obj)

                        # cache
                        CACHE[key_hash] = csv_bytes
                        save_cache(CACHE)

                        # provide downloads
                        st.download_button("Download CSV", data=csv_bytes, file_name=os.path.basename(saved_path) if saved_path else "result.csv")

                        # XLSX download (temp file)
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                            result_obj.to_excel(tmp.name, index=False)
                            tmp.seek(0)
                            st.download_button("Download XLSX", data=open(tmp.name, "rb"), file_name=(os.path.basename(saved_path).replace('.csv', '.xlsx') if saved_path else 'result.xlsx'))
                    else:
                        st.write(result_obj)
                        # cache scalar repr
                        CACHE[key_hash] = repr(result_obj).encode('utf-8')
                        save_cache(CACHE)
                except Exception as e:
                    st.error(f"Failed to display result: {e}")

st.markdown("---")
st.caption("Fast local demo. Extend SYNONYMS and column corpus building to improve retrieval quality.")
