import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.set_page_config(page_title="Book Bitch Tutor", layout="wide")
st.title("Book Bitch – Grounded Tutor")

if "book_id" not in st.session_state:
    st.session_state.book_id = None

def refresh_books() -> list[dict]:
    response = requests.get(f"{API_URL}/books", timeout=30)
    response.raise_for_status()
    return response.json()

with st.sidebar:
    st.header("Books")
    try:
        books = refresh_books()
    except requests.exceptions.RequestException:
        books = []
        st.error("Backend unavailable. Start FastAPI first.")

    display_names = [f"{book['book_id']} ({book['filename']})" for book in books]
    book_options = [book["book_id"] for book in books]

    selected_idx = None
    if st.session_state.book_id in book_options:
        selected_idx = book_options.index(st.session_state.book_id)

    selected = st.selectbox("Select a book", options=book_options, format_func=lambda x: x, index=selected_idx if selected_idx is not None else 0 if book_options else None)

    if selected:
        st.session_state.book_id = selected

    uploaded = st.file_uploader("Upload textbook (PDF)", type=["pdf"])
    if uploaded:
        file_size = getattr(uploaded, "size", None)
        if file_size is None:
            file_size = len(uploaded.getvalue())
        if file_size > MAX_FILE_SIZE_BYTES:
            st.error(f"PDF must be {MAX_FILE_SIZE_MB} MB or less.")
        elif st.button("Ingest PDF"):
            with st.spinner("Uploading and ingesting..."):
                response = requests.post(
                    f"{API_URL}/upload/",
                    files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                    timeout=600,
                )
            if response.ok:
                data = response.json()
                st.success(f"Book ingested: {data['book_id']} ({data['chunks']} chunks)")
                books = refresh_books()
                if data["book_id"] not in book_options:
                    book_options.append(data["book_id"])
                st.session_state.book_id = data["book_id"]
            else:
                st.error(response.text)

if not st.session_state.book_id:
    st.info("Upload or select a book to start chatting.")
else:
    st.subheader(f"Ask about `{st.session_state.book_id}`")
    question = st.text_area("Question")
    if st.button("Ask", use_container_width=True) and question:
        with st.spinner("Querying..."):
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "book_id": st.session_state.book_id},
                timeout=120,
            )
        if response.ok:
            data = response.json()
            st.markdown("### Answer")
            st.write(data["answer"])
            st.markdown("### Evidence")
            for idx, evidence in enumerate(data["evidence"], start=1):
                st.write(f"{idx}. Page {evidence.get('page', '?')} — {evidence.get('text', '...')}")
        else:
            st.error(response.text)