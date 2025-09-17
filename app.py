import os, json, io, faiss
import fitz, pytesseract
from PIL import Image
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
import streamlit as st

# =========================
# 🔹 CONFIG & SETUP
# =========================
DATA_DIR = os.environ.get("DATA_DIR", "jenny_pdfs")      # place PDFs here
INDEX_DIR = os.environ.get("INDEX_DIR", "jenny_index")   # place FAISS index here

SYSTEM_PROMPT = """
You are Jenny, a partner-level strategy consultant (McKinsey/BCG/Bain caliber).
ONLY answer strategy, management, markets, consulting, and business frameworks.
If the query is outside these topics (like coding, math, chit-chat, or personal),
reply strictly: "❌ I only provide Partner-level strategy & management insights."

Always apply:
- MECE structuring
- Hypothesis-driven logic
- 80/20 prioritization
- Boardroom clarity
- Quick math with assumptions

Use this output format:
1. Executive Summary
2. Client Context
3. Hypotheses
4. Key Drivers & Quick Math
5. Options & Trade-offs
6. Recommendation
7. Risks & Mitigations
8. 90-Day Action Plan
9. KPIs & Targets
10. Data Request
11. Assumptions & Confidence
"""

# =========================
# 🔹 BACKEND (Embeddings + Search + Jenny)
# =========================
# Load embeddings model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS index
index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
corpus = json.load(open(os.path.join(INDEX_DIR, "meta.json")))

def embed_texts(texts):
    return np.array(model.encode(texts, normalize_embeddings=True), dtype="float32")

def search(query, k=5):
    q = embed_texts([query])
    D, I = index.search(q, k)
    return [corpus[i] for i in I[0]]

# Jenny LLM client
GROQ_API_KEY = os.environ.get("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")  # set this in Streamlit Cloud secrets
client = Groq(api_key=GROQ_API_KEY)

PRIMARY_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_jenny(prompt, model=PRIMARY_MODEL):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1600
        )
    except Exception as e:
        if "decommissioned" in str(e).lower():
            resp = client.chat.completions.create(
                model=FALLBACK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1600
            )
        else:
            raise
    return resp.choices[0].message.content.strip()

def jenny_consult(problem, context="", use_kb=True):
    kb_text = ""
    if use_kb:
        hits = search(problem, k=5)
        kb_text = "\n\n".join([f"[{h['doc']} | p{h['page']}] {h['text'][:400]}..." for h in hits])

    user_prompt = f"""
== Problem ==
{problem}

== Context ==
{context or '(none provided)'}

== Knowledge Base ==
{kb_text}
"""
    return call_jenny(user_prompt)

# =========================
# 🔹 FRONTEND (Streamlit UI)
# =========================
st.set_page_config(
    page_title="Ask Jenny – Strategy Partner",
    page_icon="💼",
    layout="centered"
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
        .stApp { background-color: #F5F5F0; color: #0A2342; font-family: "Helvetica Neue", Arial, sans-serif; }
        h1 { font-family: "Georgia", serif; color: #0A2342; text-align: center; font-weight: 600; }
        h3 { font-family: "Georgia", serif; color: #5A5A5A; text-align: center; font-weight: 400; }
        label, .stMarkdown p { color: #0A2342 !important; font-weight: 600; font-size: 15px; }
        .stTextArea textarea { font-family: "Cookie", cursive, Georgia, serif; background-color: #FFF5EE; border: 1px solid #A7A9AC; border-radius: 8px; font-size: 15px; padding: 12px; color: #0A2342; caret-color: #0A2342; }
        div.stButton > button { background-color: #2F4156; color: #FFFFFF; border-radius: 6px; border: none; font-weight: 600; font-size: 16px; padding: 10px 24px; transition: background-color 0.3s ease; display: block; margin: 0 auto; font-family: "Georgia", serif; }
        div.stButton > button:hover { background-color: #000000; color: #ffffff; }
        .response-box { background-color: #FFFFFF; border-left: 4px solid #87CEEB; padding: 15px 20px; border-radius: 6px; margin-top: 20px; font-size: 15px; color: #0A2342; }
        .stAlert { background-color: #353839 !important; color: white !important; border-radius: 6px; font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- UI ---
st.markdown("<h1>Ask Jenny!</h1>", unsafe_allow_html=True)
st.markdown("<h3>Your Partner-level Strategy Consultant</h3>", unsafe_allow_html=True)

problem = st.text_area("Problem Statement", height=120)
context = st.text_area("Client Context", height=120)

if st.button("Ask Jenny"):
    if problem.strip() and context.strip():
        with st.spinner("Jenny is thinking..."):
            response = jenny_consult(problem, context)
        st.markdown(
            f"""
            <div class="response-box">
                <b>Jenny’s Strategic Response:</b><br><br>
                {response}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="stAlert">
                ❌ Please enter both the Problem Statement and Client Context before asking Jenny.
            </div>
            """,
            unsafe_allow_html=True
        )
