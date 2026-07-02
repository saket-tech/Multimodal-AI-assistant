"""
Multimodal AI Financial Assistant
Upload a credit card statement, invoice, receipt or expense report (image or PDF)
and ask questions about charges.
"""

import base64
import io
import json
import os

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import pdfplumber

from extractor import extract_fields
from vector_store import get_vector_store

load_dotenv()

# ── LangSmith tracing (optional) ─────────────────────────────────────────────
_ls_key = os.getenv("LANGSMITH_API_KEY", "")
if _ls_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = _ls_key
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "multimodal-financial-assistant")

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
TEXT_MODEL = "llama-3.3-70b-versatile"

client = Groq(api_key=GROQ_API_KEY)
vs = get_vector_store()

# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
    return "\n".join(text_parts)


def get_rag_context(question: str) -> str:
    """Retrieve supporting context from vector store."""
    results = vs.retrieve(question, top_k=3)
    if not results:
        return ""
    parts = [f"- {r['content']} (source: {r['source']})" for r in results]
    return "Supporting context from knowledge base:\n" + "\n".join(parts)


def analyze_image(image: Image.Image, question: str) -> str:
    """Use vision model + RAG context to analyze image and answer question."""
    img_b64 = image_to_base64(image)
    rag_context = get_rag_context(question)

    system_text = (
        "You are a financial document analyst. "
        "The user has uploaded a financial document.\n"
        "First extract key fields: vendor name, line items, amounts, dates, taxes, totals.\n"
        "Then answer the question clearly and concisely.\n"
        "Always cite specific amounts and dates from the document.\n"
    )
    if rag_context:
        system_text += f"\n{rag_context}\n"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_text + f"\nQuestion: {question}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            }
        ],
        max_tokens=1024,
    )
    return response.choices[0].message.content


def analyze_pdf_text(text: str, question: str) -> str:
    """Use text LLM + RAG context to answer question about PDF content."""
    rag_context = get_rag_context(question)

    system = (
        "You are a financial document analyst. "
        "Extract key fields and answer the user's question based only on the document. "
        "Always cite specific amounts and dates."
    )
    if rag_context:
        system += f"\n\n{rag_context}"

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Document:\n{text}\n\nQuestion: {question}"},
        ],
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="AI Financial Assistant", page_icon="🤖", layout="centered")

st.title("🤖 AI Financial Assistant")
st.caption("Upload a financial document and ask questions about any charge")

uploaded_file = st.file_uploader(
    "Upload document (image or PDF)",
    type=["png", "jpg", "jpeg", "pdf"],
    help="Supports credit card statements, invoices, receipts, expense reports",
)

if uploaded_file:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded document", use_column_width=True)
        doc_mode = "image"
        doc_text = None
    else:
        pdf_bytes = uploaded_file.read()
        doc_text = extract_pdf_text(pdf_bytes)
        st.text_area(
            "Extracted PDF text",
            doc_text[:1000] + "..." if len(doc_text) > 1000 else doc_text,
            height=200,
        )
        doc_mode = "pdf"
        image = None

    # Structured field extraction
    with st.expander("📋 Extracted Fields", expanded=False):
        with st.spinner("Extracting structured fields..."):
            try:
                if doc_mode == "pdf" and doc_text:
                    fields = extract_fields(client, TEXT_MODEL, doc_text)
                    st.json(fields)
                    # Store in vector store for future retrieval
                    vs.add_extracted_document(
                        doc_text[:2000],
                        uploaded_file.name,
                    )
                else:
                    st.info("Field extraction shown for PDF documents. For images, fields are extracted inline with each question.")
            except Exception as e:
                st.warning(f"Field extraction failed: {e}")

    st.divider()

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about a charge, e.g. 'Why was this charge deducted?'")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document..."):
                try:
                    if doc_mode == "image":
                        answer = analyze_image(image, question)
                    else:
                        if not doc_text or not doc_text.strip():
                            answer = "Could not extract text from this PDF. Try uploading an image instead."
                        else:
                            answer = analyze_pdf_text(doc_text, question)

                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    st.error(f"Error: {e}")

else:
    st.info("👆 Upload a financial document to get started")
    st.markdown("""
**What you can ask:**
- Why was this charge deducted?
- What is the total amount due?
- List all line items and amounts
- What are the taxes and fees?
- Who is the vendor for this charge?
- What was my previous balance?
    """)
