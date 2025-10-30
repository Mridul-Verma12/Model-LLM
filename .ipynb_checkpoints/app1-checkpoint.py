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
SYNONYMS = {
    # ---------------------------------
    # 1. FARMER DATASET SYNONYMS
    # ---------------------------------
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
        "Paddy": ["paddy", "basmati", "non-basmati", "parboiled rice"],
        "Wheat": ["wheat", "durum", "atta wheat", "bread wheat"],
        "Maize": ["maize", "corn", "sweetcorn", "babycorn"],
        "Millets": ["millet", "bajra", "ragi", "jowar", "sorghum", "foxtail", "barnyard"],
        "Sugarcane": ["sugarcane", "cane"],
        "Cotton": ["cotton", "bt cotton", "kapas"],
        "Pulses": ["pulse", "pulses", "dal", "lentil", "gram", "tur", "moong", "masoor", "urad"],
        "Oilseeds": ["oilseed", "mustard", "groundnut", "soybean", "sunflower", "sesame"],
        "Fruits": ["fruit", "fruits", "banana", "mango", "orange", "grape", "guava", "apple"],
        "Vegetables": ["vegetable", "vegetables", "tomato", "onion", "potato", "brinjal", "chili", "cabbage", "cauliflower"],
        "Rice": ["rice", "Rice"] # This covers both files
    },

    # ---------------------------------
    # 2. SEED DATASET SYNONYMS
    # ---------------------------------
    "crop_name": {
        "Crop Name": ["crop", "crop name", "plant", "plant name", "barley", "buckwheat", "jowar"]
        # Note: Crop names like "Wheat", "Maize", "Rice" are already in the "crop" section above
    },
    "seed_code": {
        "Seed Code": ["seed id", "seed code", "seed number", "seed_id"]
    },
    "soil_type": {
        "Soil Type": ["soil", "soil type", "ground type", "soil_type"],
        "Laterite": ["laterite"],
        "Arid": ["arid", "dry soil"],
        "Saline": ["saline", "salty soil"],
        "Loamy": ["loamy", "loam"],
        "Clayey": ["clayey", "clay"],
        "Alluvial": ["alluvial"]
    },
    "irrigation_source": {
        "Irrigation": ["irrigation", "water source", "irrigation_source"],
        "River": ["river"],
        "Drip": ["drip irrigation", "drip"],
        "Canal": ["canal"],
        "Rainfed": ["rainfed", "rain fed", "no irrigation"],
        "Sprinkler": ["sprinkler"]
    },
    "fertilizer_used": {
        "Fertilizer": ["fertilizer", "fertilizer_used", "fertiliser"],
        "Organic": ["organic"],
        "Compost": ["compost"],
        "Urea": ["urea"],
        "Potash": ["potash"],
        "NPK": ["npk"],
        "DAP": ["dap"]
    },
    "seed_type": {
        "Seed Type": ["seed type", "type of seed", "seed_type", "variety"],
        "Certified": ["certified", "certified seed"],
        "Hybrid": ["hybrid", "hybrid seed"],
        "Local": ["local", "local seed", "farmer seed", "traditional seed"],
        "F1 Hybrid": ["f1 hybrid", "f1"]
    },
    "weather_condition": {
        "Weather": ["weather", "weather condition", "weather_condition"],
        "Dry Spell": ["dry spell", "low rain"],
        "Flood": ["flood", "heavy rain"],
        "Normal": ["normal", "normal weather", "good weather"],
        "Drought": ["drought", "no rain"],
        "Excess Rain": ["excess rain"]
    },
    "ownership_type": {
        "Ownership": ["ownership", "land ownership", "ownership_type"],
        "Sharecropped": ["sharecropped", "shared land", "share cropper"],
        "Leased": ["leased", "rented", "rented land"],
        "Owned": ["owned", "own land", "owner"]
    },
    "loan_taken": {
        "Loan": ["loan", "loan taken", "loan_taken", "debt", "credit"],
        "Yes": ["yes", "taken loan", "with loan", "has loan"],
        "No": ["no", "no loan", "without loan", "no debt"]
    },
    "insurance_status": {
        "Insurance": ["insurance", "insurance status", "insurance_status", "crop insurance"],
        "Enrolled": ["enrolled", "has insurance", "insured"],
        "Not Enrolled": ["not enrolled", "no insurance", "uninsured"]
    },
    "pesticide_usage": {
        "Pesticide": ["pesticide", "pesticide usage", "pesticide_usage"],
        "Medium": ["medium", "med", "average use"],
        "Low": ["low", "little", "low use"],
        "High": [ "high", "a lot", "high use"],
        "None": ["none", "no pesticide", "zero pesticide"]
    },
    "last_sowing_date": {
        "Sowing Date": ["sowing date", "last sowing date", "last_sowing_date", "planted on"]
    },
    "expected_harvest_date": {
        "Harvest Date": ["harvest date", "expected harvest date", "expected_harvest_date", "harvest on"]
    }
}

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
                # Use filename without .csv as the key
                name = os.path.splitext(filename)[0].lower().replace(" ", "_")
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
      - ds_col_texts: mapping: dataset_name -> list of column_texts
      - ds_col_names: mapping: dataset_name -> list of column_names
    """
    corpus_texts = []
    index_map = []  # tuples (dataset, column, text)

    for ds_name, df in datasets.items():
        for col in df.columns:
            # build a small text describing the column: name + up to 5 example values
            example_vals = []
            try:
                vals = df[col].dropna().unique()
                if len(vals) > 0:
                    sample_count = min(5, len(vals))
                    if len(vals) <= 50:
                        sample_vals = [str(v) for v in vals[:sample_count]]
                    else:
                        sample_vals = [str(v) for v in np.random.choice(vals, sample_count, replace=False)]
                    example_vals = sample_vals
            except Exception:
                example_vals = []
            
            # ### --- MODIFICATION --- ###
            # Improved text: include dataset name and column name for better context
            text = f"dataset: {ds_name} column: {col} examples: " + " ".join(example_vals)
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


### --- MODIFICATION --- ###
# NEW Function: Build vectors for *entire datasets* to help select which one to use
@st.cache_resource(show_spinner=False)
def build_dataset_vectors(_vectorizer: TfidfVectorizer, _ds_col_texts: Dict[str, List[str]]):
    """
    Creates a single "document" for each dataset by joining all its column texts.
    Then transforms these documents using the *existing* vectorizer.
    
    Returns:
      - dataset_name_list: Ordered list of dataset names
      - dataset_vectors_matrix: TF-IDF matrix where each row is a dataset
    """
    if not _ds_col_texts:
        return [], None

    dataset_name_list = list(_ds_col_texts.keys())
    dataset_doc_list = []
    for ds_name in dataset_name_list:
        # Create one big text string for the entire dataset
        full_dataset_text = " ".join(_ds_col_texts[ds_name])
        dataset_doc_list.append(full_dataset_text)

    if not dataset_doc_list:
        return [], None
        
    dataset_vectors_matrix = _vectorizer.transform(dataset_doc_list)
    return dataset_name_list, dataset_vectors_matrix
### --- END MODIFICATION --- ###


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
    q = q.replace("rice farmers", "rice") # Help distinguish 'rice' as a crop

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
            # This rewrite is the key: "women" becomes "gender==Female"
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

### --- MODIFICATION --- ###
# NEW Function: Find the best *dataset* for the query
def find_top_dataset_for_query(
    query_vec: Any,
    dataset_name_list: List[str],
    dataset_vectors_matrix: Any
) -> Optional[str]:
    """
    Compares the query vector to the pre-computed dataset vectors
    and returns the name of the best-matching dataset.
    """
    if dataset_vectors_matrix is None or not dataset_name_list:
        return None
        
    sims = cosine_similarity(query_vec, dataset_vectors_matrix)[0]
    top_idx = np.argmax(sims)
    return dataset_name_list[top_idx]
### --- END MODIFICATION --- ###


### --- MODIFICATION --- ###
# Modified `find_top_columns_for_query` to accept the pre-computed query vector
def find_top_columns_for_query(
    q_vec: Any, 
    dataset_name: str, 
    vectorizer: TfidfVectorizer,
    ds_col_texts: Dict[str, List[str]], 
    ds_col_names: Dict[str, List[str]], 
    top_k=6
):
    """
    Returns list of (column_name, score, example_text) for the *selected* dataset
    """
    if dataset_name not in ds_col_texts:
        return []
    texts = ds_col_texts[dataset_name]
    names = ds_col_names[dataset_name]

    # Don't need to transform query, it's already a vector
    # q_vec = vectorizer.transform([query]) 
    
    texts_vec = vectorizer.transform(texts)
    sims = cosine_similarity(q_vec, texts_vec)[0]
    
    # Filter out very low-scoring columns
    min_score_threshold = 0.01 
    relevant_indices = np.where(sims > min_score_threshold)[0]
    
    if len(relevant_indices) == 0:
        return [] # No columns matched at all

    # Sort only the relevant indices
    sorted_relevant_indices = sorted(relevant_indices, key=lambda i: sims[i], reverse=True)
    
    top_k_indices = sorted_relevant_indices[:top_k]
    
    results = []
    for i in top_k_indices:
        results.append((names[i], float(sims[i]), texts[i]))
    return results
### --- END MODIFICATION --- ###


# -------------------------
# Prompt builder: small, structured prompt
# -------------------------

def build_minimal_prompt(user_query: str, dataset_name: str, top_cols: List[Tuple[str, float, str]], intent: str = "list"):
    cols_str = ", ".join([f"'{c[0]}'" for c in top_cols]) if top_cols else ""
    examples = "\n".join([f"- {c[0]}: (examples: {c[2]})" for c in top_cols]) if top_cols else "No relevant columns found."

    prompt = (
        "You are an expert Python pandas assistant. Respond with code only.\n"
        f"Operate on the dataframe named `datasets['{dataset_name}']`.\n"
        f"These are the most relevant columns for the query: [{cols_str}]\n"
        "Column details (with examples):\n"
        f"{examples}\n"
        ### --- MODIFICATION --- ###
        # Activated the 'intent' line you had commented out
        f"The user's intent is to: {intent}\n"
        "User request (normalized with hints like 'col==Value'):\n"
        f"{user_query}\n\n"
        "Requirements:\n"
        "- Write only Python code; do not include any explanation or markdown.\n"
        f"- Start your code with: `result = datasets['{dataset_name}'].copy()`\n"
        "- Use `result.query(...)` or pandas indexing for filtering. Use the hints in the user request (e.g., 'gender==Female').\n"
        "- If the request has hints like 'farm_area_ha==large', use the synonyms provided (e.g., 'large', 'big').\n"
        "- If filtering on string columns, be case-insensitive (e.g., `result['col'].str.contains('value', case=False)`).\n"
        "- Assign final DataFrame (or scalar value like a count) to the `result` variable.\n"
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
            options={"temperature": 0.0} # Lower temperature for more deterministic code
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
    r"sys\.",
    r"glob\."
]


def is_code_safe(code_text: str) -> Tuple[bool, Optional[str]]:
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, code_text):
            return False, f"Forbidden token or pattern found: {pat}"
    # require assignment to `result`
    if "result" not in code_text:
        return False, "No 'result' variable detected in generated code."
    # ### --- MODIFICATION --- ###
    # Block attempts to access other datasets
    if "datasets[" in code_text and not re.search(r"datasets\[['\"][^\]]+['\"]\]", code_text):
         return False, "Code attempts to access 'datasets' in an unsafe way."
    return True, None


def sanitize_code_text(code_text: str) -> str:
    if not isinstance(code_text, str):
        code_text = str(code_text)
    code_text = code_text.replace("’", "'").replace("‘", "'")
    code_text = code_text.replace("“", '"').replace("”", '"')
    code_text = code_text.encode("ascii", "ignore").decode()
    # remove fenced triples
    code_text = re.sub(r"^```(?:python)?\s*", "", code_text, flags=re.I | re.MULTILINE)
    code_text = re.sub(r"\s*```$", "", code_text, flags=re.I | re.MULTILINE)
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
    
    # Fallback: if no triple-quotes, assume the *entire* response is code
    # This is common for code-only models like phi3
    if not text.startswith("```"):
        return text.strip()
        
    return text.strip()


def safe_exec_user_code(code_text: str, datasets: dict, dataset_name: str):
    """
    Execute code in restricted environment and return (result_obj, used_code, error_str).
    """
    if not code_text:
        return None, code_text, "No code provided"
    
    # ### --- MODIFICATION --- ###
    # Sanitize *before* checking safety
    code_text = sanitize_code_text(code_text)
    
    safe, reason = is_code_safe(code_text)
    if not safe:
        return None, code_text, f"Code rejected by safety checks: {reason}"
    
    # Ensure code only references the *intended* dataset
    if "datasets[" in code_text and f"datasets['{dataset_name}']" not in code_text:
        if f'datasets["{dataset_name}"]' not in code_text:
            return None, code_text, f"Code safety check: Attempted to access a dataset other than '{dataset_name}'"

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
    
    # Detect intent
    intent = detect_intent(user_query)

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
    
    # Apply intent
    if intent == "count":
        return len(df), "applied local filters (count)"
        
    return df, "applied local filters (list)"


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
st.write("Query multiple CSVs with natural language. The app auto-detects the best CSV and columns to use.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Data")
    ensure_dirs()
    uploaded_files = st.file_uploader("Upload CSV files (e.g. farmers.csv, seeds.csv)", accept_multiple_files=True, type=["csv"])
    
    if uploaded_files:
        files_saved = 0
        for f in uploaded_files:
            try:
                path = os.path.join(DATA_DIR, f.name)
                with open(path, "wb") as out:
                    out.write(f.getbuffer())
                files_saved += 1
            except Exception as e:
                st.error(f"Error saving {f.name}: {e}")
        if files_saved > 0:
            st.success(f"Saved {files_saved} files to {DATA_DIR}")
            # ### --- MODIFICATION --- ###
            # Must clear caches when new data is uploaded
            st.cache_data.clear()
            st.cache_resource.clear()
            st.experimental_rerun()

    datasets = load_all_csvs_from_dir()

        # --- Remove uploaded CSV files ---
    st.subheader("Remove uploaded CSV")
    existing_csvs = [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    
    if existing_csvs:
        file_to_delete = st.selectbox("Select a file to delete", [""] + existing_csvs, index=0)
        if file_to_delete and st.button(f"Delete '{file_to_delete}'"):
            try:
                os.remove(os.path.join(DATA_DIR, file_to_delete))
                st.success(f"Deleted {file_to_delete}")
                # ### --- MODIFICATION --- ###
                # Must clear caches when data is deleted
                st.cache_data.clear()
                st.cache_resource.clear()
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error deleting file: {e}")
    else:
        st.info("No CSV files found in 'data/' folder.")

    if not datasets:
        st.warning("No CSVs found. Upload a CSV or add one to the 'data' folder.")
        # Create a dummy for display, but main app will be disabled
        datasets = {"(no data)": pd.DataFrame(columns=["Upload a CSV to start."])}


    st.subheader("Loaded datasets")
    for name, df in datasets.items():
        with st.expander(f"**{name}** — {len(df)} rows, {len(df.columns)} cols"):
            info_df = pd.DataFrame({"column": df.columns, "dtype": [str(dt) for dt in df.dtypes]})
            st.dataframe(info_df, use_container_width=True)

    st.markdown("---")
    st.checkbox("Show generated python code before execution", key="show_code", value=True)
    st.write(f"Using model: `{OLLAMA_MODEL}` (set MODEL_NAME variable in script to change)")

# Build vectorizer + per-dataset column corpus
# This is cached and only runs once
vectorizer, ds_col_texts, ds_col_names = build_column_corpus_and_vectorizer(datasets)

# ### --- MODIFICATION --- ###
# Build the dataset-level vectors
ds_name_list, ds_vec_matrix = build_dataset_vectors(vectorizer, ds_col_texts)
# ### --- END MODIFICATION --- ###


with col2:
    st.header("Ask a question")
    
    # Disable query box if no real data is loaded
    no_data = "(no data)" in datasets or not datasets
    
    query_examples = [
        "list female farmers who are sc and grow rice",
        "how many farmers are in karnataka",
        "show me seed prices for wheat",
        "count of marginal farmers"
    ]
    st.caption(f"e.g. `{np.random.choice(query_examples)}`")
    
    user_query = st.text_area(
        "Natural language query", 
        height=100, 
        disabled=no_data,
        placeholder="Type your query here..." if not no_data else "Upload a CSV file in the left panel to enable querying."
    )
    run = st.button("Run query", disabled=no_data)

    if run:
        if not user_query:
            st.error("Enter a query")
        else:
            
            ### --- MODIFICATION --- ###
            # This is the new pipeline
            #
            # 1) Normalize user query (lightweight)
            normalized = normalize_user_query(user_query)
            
            # 2) Vectorize the query *once*
            query_vec = vectorizer.transform([normalized])

            # 3) Find the best *dataset*
            dataset_name = find_top_dataset_for_query(query_vec, ds_name_list, ds_vec_matrix)
            if not dataset_name:
                st.error("Could not find a relevant dataset for this query.")
                st.stop()
            
            st.info(f"**Selected Dataset:** `{dataset_name}` (auto-detected)")

            # 4) Find top *columns* from that dataset
            top_cols = find_top_columns_for_query(query_vec, dataset_name, vectorizer, ds_col_texts, ds_col_names, top_k=8)

            # 5) Detect intent
            intent = detect_intent(normalized)
            
            # 6) Build prompt
            prompt = build_minimal_prompt(normalized, dataset_name, top_cols, intent)

            st.subheader("Debug Info")
            with st.expander("View generated prompt, intent, and top columns"):
                st.markdown(f"**Intent:** `{intent}`")
                st.markdown("**Top Columns:**")
                st.json({c[0]: round(c[1], 2) for c in top_cols})
                st.markdown("**Full Prompt (sent to model):**")
                st.code(prompt, language='text')

            # 7) Compute cache key
            key_obj = {
                "dataset": dataset_name,
                "normalized_query": normalized,
                "top_cols": [c[0] for c in top_cols],
                "intent": intent
            }
            key_raw = json.dumps(key_obj, sort_keys=True)
            key_hash = hashlib.sha256(key_raw.encode()).hexdigest()

            # 8) Check cache
            cached = CACHE.get(key_hash)
            if cached is not None:
                st.success("Cache hit — returning cached result")
                try:
                    # cached stored as CSV bytes
                    df_cached = pd.read_csv(pd.io.common.BytesIO(cached))
                    st.dataframe(df_cached)
                    st.download_button("Download CSV", data=cached, file_name="cached_result.csv")
                except Exception:
                    # It might be a scalar
                    st.write(cached.decode('utf-8'))
                st.stop()

            # 9) Call LLM
            with st.spinner(f"Calling model `{OLLAMA_MODEL}`..."):
                model_text, err = call_ollama(prompt)

            if err:
                st.error(f"Model call failed: {err}")
                # try local fallback
                fallback_df, reason = apply_local_simple_filters(user_query, datasets, dataset_name)
                if fallback_df is not None:
                    st.info(f"Local fallback succeeded ({reason}) — returning local result")
                    if isinstance(fallback_df, pd.DataFrame):
                        st.dataframe(fallback_df)
                        csv_bytes = fallback_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV", data=csv_bytes, file_name="result_fallback.csv")
                    else:
                        st.write(fallback_df) # scalar result
                        csv_bytes = repr(fallback_df).encode('utf-8')
                        
                    CACHE[key_hash] = csv_bytes
                    save_cache(CACHE)
                else:
                    st.error(f"Local fallback also failed: {reason}")
                st.stop()

            # 10) Extract and sanitize code
            code_candidate = extract_code_from_model(model_text)
            
            if st.session_state.get("show_code"):
                st.subheader("Generated code (sanitized)")
                st.code(code_candidate, language='python')

            # 11) Try executing safely
            result_obj, used_code, exec_err = safe_exec_user_code(code_candidate, datasets, dataset_name)

            if exec_err:
                st.error(f"Error executing generated code: {exec_err}")
                st.markdown("Model raw text:")
                st.text_area("Model raw text", value=model_text, height=150)
                
                # Attempt local fallback
                fallback_df, reason = apply_local_simple_filters(user_query, datasets, dataset_name)
                if fallback_df is not None:
                    st.info(f"Local fallback succeeded ({reason}) — returning local result")
                    if isinstance(fallback_df, pd.DataFrame):
                        st.dataframe(fallback_df)
                        csv_bytes = fallback_df.to_csv(index=False).encode("utf-8")
                        st.download_button("Download CSV", data=csv_bytes, file_name="result_fallback.csv")
                    else:
                        st.write(fallback_df) # scalar result
                        csv_bytes = repr(fallback_df).encode('utf-8')

                    CACHE[key_hash] = csv_bytes
                    save_cache(CACHE)
                else:
                    st.error(f"Local fallback also failed: {reason}")
                st.stop()

            # 12) Success — show result and cache
            if result_obj is None:
                st.warning("Model code produced no result. Showing raw model output instead.")
                st.text_area("Model raw text", value=model_text, height=150)
            else:
                st.subheader("Result")
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
                        dl_filename = os.path.basename(saved_path) if saved_path else "result.csv"
                        st.download_button("Download CSV", data=csv_bytes, file_name=dl_filename)

                        # XLSX download (temp file)
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                                result_obj.to_excel(tmp.name, index=False)
                                tmp.seek(0)
                                st.download_button(
                                    "Download XLSX", 
                                    data=open(tmp.name, "rb"), 
                                    file_name=dl_filename.replace('.csv', '.xlsx')
                                )
                        except Exception as e:
                            st.caption(f"Could not generate XLSX: {e}") # Don't fail the whole response
                    
                    else:
                        # Handle scalar results (like counts)
                        st.metric(label="Result", value=str(result_obj))
                        # cache scalar repr
                        CACHE[key_hash] = repr(result_obj).encode('utf-8')
                        save_cache(CACHE)
                        
                except Exception as e:
                    st.error(f"Failed to display result: {e}")
                    st.write(result_obj) # Fallback to raw output

st.markdown("---")
st.caption("This app uses a local LLM (Ollama) and TF-IDF semantic search to query CSVs.")