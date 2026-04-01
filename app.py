import streamlit as st
import tempfile
import os

from src.data_loader import load_pdf
from src.ocr_loader import load_image_text, load_pdf_with_ocr
from src.chunker import chunk_data
from src.embedding import get_embeddings
from src.vector_store import create_vector_db
from src.retriever import retrieve_docs
from src.groq_llm import generate_answer_groq

st.set_page_config(
    page_title="RAG PDF & Image Q&A",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    background-color: #0d1117 !important;
    font-family: 'Sora', sans-serif !important;
    color: #e6edf3 !important;
}
#MainMenu, footer, header {visibility: hidden;}
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; max-width: 780px !important; }

/* Headings */
h1, h2, h3 { font-family: 'Sora', sans-serif !important; color: #e6edf3 !important; }

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
}

/* Status cards */
.status-groq {
    background: linear-gradient(135deg, rgba(35,134,54,0.1), rgba(35,134,54,0.03));
    border: 1px solid rgba(35,134,54,0.25);
    border-radius: 14px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
}
.status-success {
    background: linear-gradient(135deg, rgba(35,134,54,0.12), rgba(35,134,54,0.03));
    border: 1px solid rgba(35,134,54,0.3);
    border-radius: 14px;
    padding: 18px 22px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 24px;
}
.s-icon {
    width: 42px; height: 42px; min-width: 42px;
    background: rgba(35,134,54,0.2);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.s-title { font-size: 14px; font-weight: 600; color: #3fb950; }
.s-sub   { font-size: 12px; color: #8b949e; margin-top: 3px; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #0d1117 !important;
    border: 2px dashed #30363d !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: #388bfd !important; }
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    color: #8b949e !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #388bfd !important;
    color: #fff !important;
    border: none !important;
    border-radius: 20px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover { background: #1f6feb !important; }
[data-testid="stFileUploader"] section { padding: 8px !important; }
[data-testid="stFileUploader"] label { color: #e6edf3 !important; font-weight: 600 !important; font-size: 15px !important; }

/* Uploaded file pills */
[data-testid="stFileUploaderFile"] {
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderFileName"] { color: #e6edf3 !important; font-size: 13px !important; }
[data-testid="stFileUploaderFileData"] { color: #8b949e !important; font-size: 11px !important; }

/* Text area */
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 14px !important;
    caret-color: #388bfd;
}
.stTextArea textarea:focus {
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(56,139,253,0.12) !important;
}
.stTextArea label { color: #e6edf3 !important; font-weight: 600 !important; font-size: 15px !important; }

/* Button */
.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100%;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2ea043 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #8b949e !important;
    font-size: 13px !important;
    font-family: 'Sora', sans-serif !important;
}

/* Divider */
hr { border-color: #21262d !important; }

/* Spinner */
[data-testid="stSpinner"] { color: #388bfd !important; }

/* Info / warning / error */
[data-testid="stAlert"] {
    background: #161b22 !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 13px !important;
}
</style>
""", unsafe_allow_html=True)

# ── HEADER ──────────────────────────────────────────────
col_a, col_b = st.columns([1, 1])
with col_a:
    st.markdown("### ✦ RAG PDF & Image Q&A")
with col_b:
    st.markdown("<p style='text-align:right;color:#8b949e;font-size:12px;font-family:JetBrains Mono,monospace;margin-top:8px'>v2.0 · Groq LLM</p>", unsafe_allow_html=True)

st.markdown("<hr style='margin:8px 0 24px'>", unsafe_allow_html=True)

# ── GROQ STATUS ──────────────────────────────────────────
st.markdown("""
<div class="status-groq">
  <div class="s-icon">⚡</div>
  <div>
    <div class="s-title">Using Groq LLM (Fast Response)</div>
    <div class="s-sub">Ultra-fast inference with state-of-the-art accuracy</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── UPLOAD ───────────────────────────────────────────────
st.markdown("**Upload Documents**")
uploaded_files = st.file_uploader(
    "Upload Documents",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)
st.markdown("<p style='font-size:11px;color:#6e7681;margin-top:6px'>Supported: PDF, PNG, JPG · Max 200MB per file</p>", unsafe_allow_html=True)

# ── HELPERS ──────────────────────────────────────────────
def is_scanned_pdf(docs):
    WATERMARKS = ["scanned by camscanner", "www.camscanner.com"]
    total = " ".join([d.page_content for d in docs]).lower()
    for w in WATERMARKS:
        total = total.replace(w, "")
    return len(total.strip()) < 50

def process_files(files):
    all_docs = []
    for file in files:
        suffix = file.name.split(".")[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        try:
            if suffix == "pdf":
                docs = load_pdf(tmp_path)
                if not docs or is_scanned_pdf(docs):
                    st.info(f"📷 '{file.name}' appears scanned — using OCR...")
                    docs = load_pdf_with_ocr(tmp_path)
            else:
                from langchain_core.documents import Document
                text = load_image_text(tmp_path)
                docs = [Document(page_content=text, metadata={"source": file.name})] if text.strip() else []
            if docs:
                all_docs.extend(docs)
            else:
                st.warning(f"⚠️ No text extracted from '{file.name}'.")
        except Exception as e:
            st.error(f"Error processing '{file.name}': {e}")
        finally:
            os.remove(tmp_path)
    return all_docs

# ── PROCESS & Q&A ─────────────────────────────────────────
if uploaded_files:
    with st.spinner("Processing documents..."):
        all_docs = process_files(uploaded_files)
        if not all_docs:
            st.error("❌ No text could be extracted.")
            st.stop()
        chunks    = chunk_data(all_docs)
        embeddings = get_embeddings()
        vector_db  = create_vector_db(chunks, embeddings)

    st.markdown("<div style='margin-top:20px'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="status-success">
      <div class="s-icon">✓</div>
      <div>
        <div class="s-title">Documents processed successfully</div>
        <div class="s-sub">{len(chunks)} chunks indexed and ready</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin:4px 0 24px'>", unsafe_allow_html=True)

    # ── QUESTION BOX ──
    st.markdown("**Ask Questions or Give Instructions**")
    st.markdown("<p style='font-size:12px;color:#8b949e;margin-bottom:10px'>Ask questions about your documents or tell the system what you need</p>", unsafe_allow_html=True)

    query = st.text_area(
        "question",
        placeholder="Type your question or instruction here...\n\nExamples:\n• What are the main points discussed in the document?\n• Summarize the key findings in 100 words\n• Extract all dates mentioned in the file",
        height=170,
        label_visibility="collapsed"
    )

    st.markdown("<p style='font-size:11px;color:#6e7681;margin-bottom:12px'>Press <b>Ctrl + Enter</b> to submit</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col3:
        send = st.button("➤ Send")

    # ── ANSWER ──
    if send and query.strip():
        with st.spinner("Generating answer..."):
            docs   = retrieve_docs(vector_db, query)
            answer = generate_answer_groq(query, docs)

        st.markdown("<div style='margin-top:24px'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card">
          <p style="font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#388bfd;font-family:'JetBrains Mono',monospace;margin-bottom:12px">ANSWER</p>
          <p style="font-size:14px;line-height:1.8;color:#e6edf3">{answer}</p>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🔍 View Retrieved Context"):
            for i, doc in enumerate(docs):
                st.markdown(f"""
                <div style="background:#0d1117;border:1px solid #21262d;border-left:3px solid #388bfd;
                     border-radius:8px;padding:12px 16px;margin-bottom:10px;
                     font-size:12px;color:#8b949e;font-family:'JetBrains Mono',monospace;line-height:1.7">
                  <span style="color:#388bfd;font-weight:600;font-size:10px;text-transform:uppercase;letter-spacing:0.08em">
                    Chunk {i+1}
                  </span><br><br>{doc.page_content}
                </div>
                """, unsafe_allow_html=True)