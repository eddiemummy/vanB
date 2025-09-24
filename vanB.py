import os
import io
import base64
import tempfile
import streamlit as st
from typing import List, Tuple, Optional

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_ollama.chat_models import ChatOllama

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

APP_NAME = "Ludvig Van Beethoven"
INDEX_DIR = "./data/index/faiss"
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)

st.set_page_config(page_title=f"🎼 {APP_NAME} – RAG Chat", layout="wide")
st.title(f"🎼 {APP_NAME}")
st.caption("📚 Dosya/Resim yükle, indeksle ve RAG ile sorular sor.")

with st.sidebar:
    st.subheader("⚙️ Ayarlar")
    temperature = st.slider("Sıcaklık", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Retriever k (kaç parça getirilsin)", 2, 10, 4, 1)
    chunk_size = st.number_input("Chunk size", 256, 4000, 1200, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 800, 200, 10)

    st.markdown("---")
    st.markdown("### 🔐 API Anahtarları")
    openai_key = st.text_input("OPENAI_API_KEY", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    google_key = st.text_input("GOOGLE_API_KEY", type="password", value=os.getenv("GOOGLE_API_KEY", ""))

    st.markdown("---")
    st.markdown("### 📦 Vektör İndeksi")
    reset_index = st.button("İndeksi Sıfırla (Sil)")
    if reset_index:
        try:
            for root, dirs, files in os.walk(INDEX_DIR, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            if os.path.exists(INDEX_DIR):
                os.rmdir(INDEX_DIR)
            st.success("İndeks silindi.")
        except Exception as e:
            st.error(f"Silinirken hata: {e}")

def load_llm() -> object:
        return ChatOllama(model="gpt-oss:20b", temperature=0.0)

def load_documents_from_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
    """PDF/TXT/DOCX/IMG dosyalarından Document listesi üretir."""
    docs: List[Document] = []
    for f in uploaded_files:
        file_bytes = f.read()
        suffix = os.path.splitext(f.name)[1].lower()
        path = os.path.join(UPLOAD_DIR, f.name)
        with open(path, "wb") as out:
            out.write(file_bytes)

        if suffix in [".pdf"]:
            loader = PyPDFLoader(path)
            ld = loader.load()
            for d in ld:
                d.metadata = d.metadata or {}
                d.metadata.update({"source": f.name})
            docs.extend(ld)

        elif suffix in [".txt", ".md", ".csv", ".py", ".json"]:
            loader = TextLoader(path, encoding="utf-8")
            ld = loader.load()
            for d in ld:
                d.metadata = d.metadata or {}
                d.metadata.update({"source": f.name})
            docs.extend(ld)

        elif suffix in [".docx"]:
            loader = Docx2txtLoader(path)
            ld = loader.load()
            for d in ld:
                d.metadata = d.metadata or {}
                d.metadata.update({"source": f.name})
            docs.extend(ld)

        elif suffix in [".png", ".jpg", ".jpeg", ".webp"]:
            text = ""
            if OCR_AVAILABLE:
                try:
                    image = Image.open(io.BytesIO(file_bytes))
                    text = pytesseract.image_to_string(image)
                except Exception as e:
                    st.warning(f"OCR hatası ({f.name}): {e}")
            else:
                st.info("OCR etkin değil (pytesseract yüklü değil). Görsellerden metin çıkarmak için pytesseract kurun.")

            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": f.name, "type": "image_ocr"}))
        else:
            st.warning(f"Desteklenmeyen dosya türü: {f.name}")

    return docs

def build_or_update_index(all_docs: List[Document], chunk_size=1200, chunk_overlap=200) -> Optional[FAISS]:
    """Belge listesiyle FAISS indeksini oluşturur veya günceller."""
    if not all_docs:
        return None

    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(INDEX_DIR):
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            vs.add_documents(chunks)
            vs.save_local(INDEX_DIR)
            return vs
        except Exception:
            pass

    vs = FAISS.from_documents(chunks, embeddings)
    vs.save_local(INDEX_DIR)
    return vs

def load_index() -> Optional[FAISS]:
    if not os.path.exists(INDEX_DIR):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    try:
        return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception:
        return None

def make_system_prompt() -> str:
    return (
        f"Sen {APP_NAME} adlı bir yardımcı asistansın. "
        "Kullanıcının yüklediği dökümanlardan ilgili parçaları al ve kaynak göstererek yanıt ver. "
        "Cevabın kısa, doğru ve Türkçe olsun. Kaynakları sonunda madde halinde listele."
    )

def rag_answer(llm, vectorstore: FAISS, query: str, k: int = 4) -> Tuple[str, List[Document]]:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    context = "\n\n---\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    sys = make_system_prompt()
    prompt = (
        f"{sys}\n\n"
        f"KULLANICI SORUSU:\n{query}\n\n"
        f"BAĞLAM (en alakalı parçalar):\n{context}\n\n"
        "Yanıt formatı:\n"
        "- Kısa bir cevap paragrafı\n"
        "- Kaynaklar: [1] DosyaAdı, [2] ... şeklinde\n"
    )
    resp = llm.invoke(prompt)
    return resp.content, docs

def render_sources(docs: List[Document]):
    if not docs:
        return
    st.markdown("**Kaynaklar:**")
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "bilinmiyor")
        st.write(f"[{i}] {src}")

st.subheader("📥 Dosya/Resim Yükle")
uploads = st.file_uploader(
    "PDF, TXT, DOCX, PNG, JPG ekleyebilirsin (birden fazla seç)",
    type=["pdf", "txt", "md", "csv", "py", "json", "docx", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

col_a, col_b = st.columns(2)
with col_a:
    if st.button("🧱 İndeksi Güncelle / Oluştur"):
        with st.spinner("İndeks güncelleniyor..."):
            new_docs = load_documents_from_files(uploads) if uploads else []
            if not new_docs and not os.path.exists(INDEX_DIR):
                st.warning("Önce en az bir dosya yükleyin.")
            else:
                vs = build_or_update_index(new_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if vs:
                    st.success("İndeks hazır!")
                else:
                    st.info("Yeni eklenecek içerik bulunamadı.")

with col_b:
    vs_test = load_index()
    st.info("İndeks durumu: " + ("✅ Yüklü" if vs_test else "❌ Yok"))

st.markdown("---")

st.subheader("💬 Soru Sor (RAG)")
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

query = st.chat_input("Sorunu yaz veya önce dosya ekleyip indeksle…")
if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append(("user", query))

    vs = load_index()
    if not vs:
        with st.chat_message("assistant"):
            st.warning("Önce en az bir dosya yükleyip **İndeksi Güncelle** demelisin.")
    else:
        llm = load_llm()
        with st.chat_message("assistant"):
            with st.spinner("Yanıt hazırlanıyor..."):
                try:
                    answer, used_docs = rag_answer(llm, vs, query, k=top_k)
                    st.markdown(answer)
                    render_sources(used_docs)
                    st.session_state.messages.append(("assistant", answer))
                except Exception as e:
                    st.error(f"Hata: {e}")
