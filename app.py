import os, io, zipfile, glob, re, json, requests
import streamlit as st
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import yaml
from pptx import Presentation
from pptx.util import Pt, Inches
from pptx.dml.color import RGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from docx import Document
from docx.shared import Pt as DOCXPt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import getpass

# ==============================
# Directories
# ==============================
DATA_DIR   = os.environ.get("DATA_DIR", "jenny_pdfs")    # PDFs
INDEX_DIR  = os.environ.get("INDEX_DIR", "jenny_index")  # Indices
YAML_DIR   = os.environ.get("YAML_DIR", "JENNY_YAML")   # YAMLs

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(YAML_DIR, exist_ok=True)

# ==============================
# Load Groq API Key
# ==============================
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Paste your Groq API key (hidden): ").strip()

# ==============================
# Corpus backend
# ==============================
@dataclass
class CorpusIndex:
    docs: List[str]
    vectorizer: Optional[TfidfVectorizer]
    matrix: Optional[np.ndarray]
    sources: List[Tuple[str, int]]  # (filename, chunk_id)

def extract_text_from_pdf(file_path: str) -> str:
    parts = []
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                parts.append(page.get_text("text") or "")
    except Exception as e:
        print(f"⚠️ PDF read error for {file_path}: {e}")
    return "\n".join(parts)

def chunk_text(text: str, max_chars: int = 1800, overlap: int = 220) -> List[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        chunks.append(text[i:end])
        i = end - overlap if end - overlap > i else end
    return chunks

def build_index(pdf_dir: str) -> CorpusIndex:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    all_chunks, sources = [], []
    for p in pdf_paths:
        raw = extract_text_from_pdf(p)
        chs = chunk_text(raw)
        base = os.path.basename(p)
        for idx, ch in enumerate(chs):
            all_chunks.append(ch)
            sources.append((base, idx))
    if not all_chunks:
        all_chunks, sources = ["(No docs indexed)"], [("—", 0)]
    vec = TfidfVectorizer(min_df=1, max_df=0.95)
    mat = vec.fit_transform(all_chunks)
    print(f"✅ Indexed {len(all_chunks)} chunks from {len(set(b for b,_ in sources))} PDFs.")
    return CorpusIndex(all_chunks, vec, mat, sources)

def retrieve(corpus: CorpusIndex, query: str, k: int = 5):
    if corpus is None or corpus.vectorizer is None or corpus.matrix is None:
        return []
    qv = corpus.vectorizer.transform([query])
    sims = cosine_similarity(qv, corpus.matrix).ravel()
    topk = sims.argsort()[::-1][:k]
    return [(corpus.docs[i], sims[i]) for i in topk]

# ==============================
# Groq backend
# ==============================
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_API_URL = f"{GROQ_BASE}/chat/completions"
GROQ_MODELS_URL = f"{GROQ_BASE}/models"

SYSTEM_PROMPT = (
    "You are Jenny, a senior strategy consultant. STRICTLY business/strategy only. "
    "Answer in crisp MECE style with **bold section headings** (~300–400 words per section)."
)

BUSINESS_KEYWORDS = [
    "strategy","market","pricing","cost","profit","revenue","growth","gtm","segment",
    "competitive","positioning","roi","wacc","valuation","okrs","kpi","unit economics",
    "capex","opex","churn","retention","forecast","five forces","blue ocean","minto",
    "scqa","sales","marketing","product","operations","supply chain","finance","saas",
    "fintech","healthcare","ecommerce","logistics","manufacturing","ai","telecom","energy","ev"
]

PREFERRED_MODELS = ["llama-3.3-70b-versatile","llama-3.1-8b-instant"]

def _groq_headers():
    key = os.environ.get("GROQ_API_KEY","").strip()
    if not key: raise RuntimeError("GROQ_API_KEY not set.")
    return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

def _groq_list_models(timeout_s: int = 30) -> list[str]:
    try:
        r = requests.get(GROQ_MODELS_URL, headers=_groq_headers(), timeout=timeout_s)
        if r.status_code != 200: return []
        data = r.json()
        return [m.get("id") for m in data.get("data", []) if m.get("id")]
    except: return []

def _pick_model() -> str:
    available = _groq_list_models()
    for m in PREFERRED_MODELS:
        if m in available: return m
    for m in available:
        if "llama" in m: return m
    return "llama-3.1-8b-instant"

def _call_groq(prompt: str, temperature: float = 0.2, max_tokens: int = 3000):
    model = _pick_model()
    payload = {
        "model": model,
        "messages": [{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    try:
        resp = requests.post(GROQ_API_URL, headers=_groq_headers(), data=json.dumps(payload), timeout=90)
        if resp.status_code != 200: return None
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()
    except: return None

def jenny_answer(question: str, corpus: CorpusIndex):
    if not any(k in (question or "").lower() for k in BUSINESS_KEYWORDS):
        return "**Refusal — Non-Business Query**\n- Only business/strategy questions allowed."
    try:
        hits = retrieve(corpus, question, k=5) if corpus else []
        ctx = "\n".join([h[0] for h in hits])[:2400] if hits else "(no context)"
    except: ctx = "(no context)"
    user_prompt = f"**Question**: {question}\n**Context**:\n{ctx}"
    ans = _call_groq(user_prompt)
    return ans if ans else "(Jenny fallback answer — no response from Groq)"

# ==============================
# Exports (PPTX, DOCX, XLSX, ZIP)
# ==============================
def make_pptx(ans: str, q: str) -> bytes:
    prs = Presentation()
    blank = prs.slide_layouts[6]
    NAVY = RGBColor(18,28,48); GREY = RGBColor(90,98,110)
    def add_slide(title, body):
        s = prs.slides.add_slide(blank)
        tf = s.shapes.add_textbox(Inches(0.7), Inches(0.6), Inches(8.4), Inches(0.9)).text_frame
        p = tf.paragraphs[0]; p.text = title; p.font.size = Pt(32); p.font.bold=True; p.font.color.rgb=NAVY
        bx = s.shapes.add_textbox(Inches(0.7), Inches(2.1), Inches(8.4), Inches(4.8)).text_frame
        bx.word_wrap=True; bx.clear()
        for line in [l.strip() for l in body.splitlines() if l.strip()]:
            para = bx.add_paragraph(); para.text=line; para.font.size=Pt(18)
    add_slide(f"Jenny — {q}", ans)
    bio = io.BytesIO(); prs.save(bio); bio.seek(0)
    return bio.read()

def make_docx(ans: str, q: str) -> bytes:
    doc = Document()
    doc.add_paragraph(f"Jenny — Strategy Memo", style='Title')
    doc.add_paragraph(q)
    doc.add_paragraph(ans)
    bio = io.BytesIO(); doc.save(bio); bio.seek(0)
    return bio.read()

def make_excel() -> bytes:
    wb = Workbook(); ws = wb.active; ws.title="Sample"
    ws.append(["Metric","Value"]); ws.append(["Example",123])
    bio=io.BytesIO(); wb.save(bio); bio.seek(0)
    return bio.read()

# ==============================
# Streamlit UI
# ==============================
st.markdown("<h1 style='text-align:center;color:#0A2342'>JENNY</h1>", unsafe_allow_html=True)
problem = st.text_area("Problem Statement", height=120)
context = st.text_area("Client Context", height=120)

corpus = build_index(DATA_DIR)

if st.button("Ask Jenny"):
    if problem.strip() and context.strip():
        with st.spinner("Jenny is thinking..."):
            answer = jenny_answer(problem, corpus)
        st.markdown(f"<div style='background:#FFFDF7;border-left:6px solid #D4AF37;padding:20px;border-radius:12px'>{answer}</div>", unsafe_allow_html=True)

        # EXPORT
        col1,col2,col3,col4 = st.columns(4)
        with col1:
            st.download_button("📊 PPT", data=make_pptx(answer, problem), file_name="jenny.pptx")
        with col2:
            st.download_button("📄 DOCX", data=make_docx(answer, problem), file_name="jenny.docx")
        with col3:
            st.download_button("📈 XLSX", data=make_excel(), file_name="jenny.xlsx")
        with col4:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as z:
                z.writestr("jenny.pptx", make_pptx(answer, problem))
                z.writestr("jenny.docx", make_docx(answer, problem))
                z.writestr("jenny.xlsx", make_excel())
            zip_buffer.seek(0)
            st.download_button("🗂 ZIP", data=zip_buffer, file_name="jenny_artifacts.zip")
    else:
        st.error("❌ Please enter both Problem Statement and Client Context.")
