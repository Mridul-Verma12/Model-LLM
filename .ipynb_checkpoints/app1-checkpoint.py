import os
import difflib
import re
import json
import pandas as pd
import streamlit as st

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

st.set_page_config(page_title="Farmer Data Assistant", layout="wide")
st.title("🌾 Farmer Data Assistant — Chat")

st.markdown(
    "Upload a CSV and ask natural-language queries like:\n"
    "- 'List scheduled caste farmers'\n"
    "- 'Show female maize growers'\n"
    "- 'How many OBC farmers grow rice?'\n"
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    os.makedirs("data", exist_ok=True)
    save_path = os.path.join("data", uploaded.name)
    with open(save_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved to {save_path}")

csv_files = [f for f in os.listdir("data") if f.endswith(".csv")] if os.path.exists("data") else []
if not csv_files:
    st.info("Please upload a CSV to begin.")
    st.stop()

selected = st.selectbox("Select dataset", csv_files)
df = pd.read_csv(os.path.join("data", selected)).fillna("")
df.columns = [col.strip() for col in df.columns]
st.write(f"Loaded `{selected}` — {len(df)} rows × {len(df.columns)} columns")

synonym_map = {
    "scheduled caste": "SC", "sc": "SC",
    "scheduled tribe": "ST", "st": "ST",
    "other backward caste": "OBC", "obc": "OBC",
    "general": "General",
    "female": "Female", "woman": "Female", "lady": "Female", "girl": "Female",
    "male": "Male", "man": "Male", "boy": "Male",
    "paddy": "Rice", "rice": "Rice", "corn": "Maize", "millet": "Jowar",
    "sugar cane": "Sugarcane", "ground nut": "Groundnut", "soy": "Soybean", "soyabean": "Soybean"
}

generic_tokens = {"list", "down", "show", "display", "farmers", "farmer", "how", "many", "number", "of"}

def normalize(text):
    return str(text).strip().lower()

def fuzzy_best(query, choices, cutoff=0.85):
    q = query.lower()
    choices_low = [c.lower() for c in choices]
    matches = difflib.get_close_matches(q, choices_low, n=1, cutoff=cutoff)
    if matches:
        idx = choices_low.index(matches[0])
        return choices[idx]
    return None

value_index = {}
for col in df.columns:
    uniq = list(df[col].dropna().astype(str).unique())
    value_index[col] = {u.lower(): u for u in uniq}

def match_values_with_synonyms(col, tokens):
    matched = set()
    vals_map = value_index.get(col, {})
    for t in tokens:
        if t in generic_tokens:
            continue
        tnorm = synonym_map.get(t.lower(), t.lower())
        if tnorm in vals_map:
            matched.add(vals_map[tnorm])
            continue
        fuzzy = fuzzy_best(tnorm, list(vals_map.keys()), cutoff=0.85)
        if fuzzy:
            matched.add(vals_map[fuzzy])
            continue
        for v in vals_map:
            if tnorm in v or v in tnorm:
                matched.add(vals_map[v])
    return sorted(list(matched))

def extract_tokens(text):
    text = normalize(text)
    tokens = set()
    for phrase in sorted(synonym_map.keys(), key=lambda x: -len(x.split())):
        if phrase in text:
            tokens.add(phrase)
            text = text.replace(phrase, "")
    for m in re.findall(r"[a-zA-Z]{2,}", text):
        if m not in generic_tokens:
            tokens.add(m.strip())
    return list(tokens)

def parse_intent_with_llama3(user_query):
    if not OLLAMA_AVAILABLE:
        return {"intent": "list", "filters": [], "return_columns": []}
    prompt = f"""
    You are a data assistant intent parser. Output ONLY valid JSON with keys:
    {{
     "intent": "list" | "count" | "chart" | "summary",
     "filters": [ {{"column": "<approx column name or null>", "values": ["v1","v2"], "op":"in"}} ],
     "return_columns": ["col1","col2"]
    }}
    Map generic user words (e.g. 'women', 'birth date', 'other backward caste') to likely column/value names (do not invent values).
    User query: {user_query}
    """
    try:
        response = ollama.chat(model="llama3-local", messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
        content = response.get("message", {}).get("content", "")
        start = content.find("{")
        end = content.rfind("}")
        return json.loads(content[start:end+1])
    except Exception:
        return {"intent": "list", "filters": [], "return_columns": []}

st.markdown("### Ask a question")
user_q = st.text_input("Type your query and press Enter", key="user_q")

if user_q and st.button("Send"):
    with st.spinner("Processing..."):
        parsed = parse_intent_with_llama3(user_q)
        intent = parsed.get("intent", "list")
        return_cols = parsed.get("return_columns", [])
        filters = parsed.get("filters", [])

        tokens = extract_tokens(user_q)
        matched_filters = {}
        for f in filters:
            col_hint = f.get("column")
            vals = f.get("values", [])
            if col_hint:
                col_match = fuzzy_best(col_hint, df.columns, cutoff=0.7)
                if col_match:
                    mv = match_values_with_synonyms(col_match, vals)
                    if mv:
                        matched_filters[col_match] = mv

        result_df = df.copy()
        for col, vals in matched_filters.items():
            result_df = result_df[result_df[col].astype(str).str.lower().isin([v.lower() for v in vals])]

        display_cols = []
        for rc in return_cols:
            mc = fuzzy_best(rc, df.columns, cutoff=0.7)
            if mc and mc not in display_cols:
                display_cols.append(mc)
        if not display_cols:
            display_cols = list(df.columns)

        st.subheader("Matched Filters")
        if matched_filters:
            st.json(matched_filters)
        else:
            st.write("No filters matched. Showing full dataset.")

        st.subheader("Results")
        st.write(f"Rows matched: {len(result_df)}")
        st.dataframe(result_df[display_cols].head(500))

        csv_bytes = result_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv_bytes, file_name="filtered_results.csv")

st.markdown("---")
st.caption("This assistant uses Llama3 via Ollama, synonym and fuzzy matching to interpret your queries. No data is stored.")