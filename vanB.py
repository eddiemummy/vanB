import os
import io
import base64
import tempfile
import streamlit as st
from typing import List, Tuple, Optional

APP_NAME = "Ludvig Van Beethoven"
INDEX_DIR = "./data/index/faiss"
UPLOAD_DIR = "./data/uploads"
DEFAULT_TEMPERATURE = 0.2

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(INDEX_DIR), exist_ok=True)

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.schema import Document

from langchain_ollama.chat_models import ChatOllama

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False


st.set_page_config(page_title=f"🎼 {APP_NAME} – RAG Chat", layout="wide")
st.title(f"🎼 {APP_NAME}")
st.caption("📚 Dosya/Resim yükle, indeksle ve RAG ile sorular sor. İndeks yoksa normal sohbet başlar.")

with st.sidebar:
    st.subheader("⚙️ Ayarlar")
    top_k = st.slider("Retriever k (kaç parça getirilsin)", 2, 10, 4, 1)
    chunk_size = st.number_input("Chunk size", 256, 4000, 1200, 50)
    chunk_overlap = st.number_input("Chunk overlap", 0, 800, 200, 10)

    st.markdown("---")
    st.markdown("### 📦 Vektör İndeksi")
    reset_index = st.button("İndeksi Sıfırla (Sil)")
    if reset_index:
        try:
            if os.path.exists(INDEX_DIR):
                for item in os.listdir(INDEX_DIR):
                    item_path = os.path.join(INDEX_DIR, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
            st.success("İndeks içeriği silindi.")
        except Exception as e:
            st.error(f"Silinirken hata: {e}")

    if st.button("💬 Sohbeti Sıfırla"):
        st.session_state.messages = []
        st.rerun()

# DEĞİŞİKLİK 2: get_embeddings fonksiyonu OllamaEmbeddings kullanacak şekilde güncellendi
def get_embeddings():
    # RAG için popüler ve ücretsiz Ollama gömme modeli kullanılır.
    # Lütfen Ollama'da bu modelin kurulu olduğundan emin olun: 'ollama pull nomic-embed-text'
    model_name = "nomic-embed-text" 
    
    # OllamaEmbeddings ile başlatma
    ollama_embeddings = OllamaEmbeddings(
        model=model_name
    )
    return ollama_embeddings

def load_llm() -> object:
    temp = DEFAULT_TEMPERATURE 
    return ChatOllama(model="gpt-oss:20b", temperature=temp)

def load_documents_from_files(uploaded_files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> List[Document]:
    """PDF/TXT/DOCX/IMG dosyalarından Document listesi üretir."""
    docs: List[Document] = []
    for f in uploaded_files:
        file_bytes = f.read()
        suffix = os.path.splitext(f.name)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file_bytes)
            path = tmp_file.name

        try:
            if suffix in [".pdf"]:
                loader = PyPDFLoader(path)
            elif suffix in [".txt", ".md", ".csv", ".py", ".json"]:
                loader = TextLoader(path, encoding="utf-8")
            elif suffix in [".docx"]:
                loader = Docx2txtLoader(path)
            else:
                loader = None

            if loader:
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
                    st.info("OCR etkin değil yani pytesseract yüklü değil. Görsellerden metin çıkarmak için pytesseract kurulmalı.")

                if text.strip():
                    docs.append(Document(page_content=text, metadata={"source": f.name, "type": "image_ocr"}))
            else:
                st.warning(f"Desteklenmeyen dosya türü: {f.name}")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    return docs

def build_or_update_index(all_docs: List[Document], chunk_size=1200, chunk_overlap=200) -> Optional[FAISS]:
    if not all_docs and not os.path.exists(INDEX_DIR):
        return None

    splitter = CharacterTextSplitter(separator="\n", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(all_docs)
    
    embeddings = get_embeddings()

    if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
        try:
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            if chunks:
                vs.add_documents(chunks)
                vs.save_local(INDEX_DIR)
            return vs
        except Exception:
            if not chunks:
                 return None
            pass

    if chunks:
        # FAISS.afrom_documents yerine FAISS.from_documents kullanıldı (async olmadığı için)
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(INDEX_DIR)
        return vs
        
    return None

def load_index() -> Optional[FAISS]:
    if not os.path.exists(INDEX_DIR) or not os.listdir(INDEX_DIR):
        return None
    embeddings = get_embeddings()
    try:
        if os.path.exists(INDEX_DIR) and os.listdir(INDEX_DIR):
            return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        return None
    except Exception:
        return None

def make_system_prompt(is_rag: bool) -> str:
    if is_rag:
        return (
            f"Sen {APP_NAME} adlı bir yardımcı asistansın. "
            "Kullanıcının yüklediği dökümanlardan ilgili parçaları al ve kaynak göstererek yanıt ver. "
            "Cevabın kısa, doğru ve Türkçe olsun. Kaynakları sonunda madde halinde listele. "
            "Eğer bağlamda soruyla ilgili bilgi bulamıyorsan, genel bilgi ile cevap ver ve 'Kaynaklar:' kısmını ekleme."
        )
    else:
        return (
            f"Sen {APP_NAME} adlı bir yardımcı asistansın. "
            "Kısa, doğru ve akıcı Türkçe cevaplar ver. Bağlamda bilgi bulamadığın için genel bilgi ile yanıt veriyorsun."
        )

def format_history_for_prompt(messages: List[Tuple[str, str]]) -> str:
    """Streamlit mesajlarını LLM prompt'u için formatlar."""
    history = []
    for role, content in messages:
        if role == "user":
            history.append(f"Kullanıcı: {content}")
        elif role == "assistant":
            clean_content = content.split('\n**Kaynaklar:**')[0].strip()
            history.append(f"Asistan: {clean_content}")
    
    return "\n".join(history[:-1])

def rag_answer(llm, vectorstore: FAISS, query: str, history: str, k: int = 4) -> Tuple[str, List[Document]]:
    """RAG kullanarak cevap üretir."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = "\n\n---\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(docs)])
    sys = make_system_prompt(is_rag=True)

    prompt = (
        f"{sys}\n\n"
        f"KONUŞMA GEÇMİŞİ:\n{history}\n\n"
        f"BAĞLAM (en alakalı parçalar):\n{context}\n\n"
        f"KULLANICI SORUSU:\n{query}\n\n"
        "Yanıt formatı:\n"
        "- Kısa ve dökümanlara dayalı bir cevap paragrafı\n"
        "- Eğer döküman kullanıldıysa: Kaynaklar: [1] DosyaAdı, [2] ... şeklinde liste\n"
        "NOT: Eğer bağlam alakasızsa veya boşsa, dökümanlara dayalı olmadan cevap ver ve kaynak gösterme."
    )
    resp = llm.invoke(prompt)
    return resp.content, docs

def chat_answer(llm, query: str, history: str) -> str:
    """Sadece chat geçmişini kullanarak cevap üretir (RAG yokken)."""
    sys = make_system_prompt(is_rag=False)
    
    prompt = (
        f"{sys}\n\n"
        f"KONUŞMA GEÇMİŞİ:\n{history}\n\n"
        f"KULLANICI SORUSU:\n{query}\n\n"
        "Kısa ve ilgili cevabını ver."
    )
    resp = llm.invoke(prompt)
    return resp.content

def render_sources(docs: List[Document]):
    if not docs:
        return
    unique_sources = {}
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "bilinmiyor")
        unique_sources[src] = unique_sources.get(src, []) + [i]

    st.markdown("**Kaynaklar:**")
    for src in unique_sources:
        st.write(f"- {src}")

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
            vs = build_or_update_index(new_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
            
            if new_docs:
                st.success(f"İndeks hazır! {len(new_docs)} yeni belge eklendi/güncellendi.")
            elif vs:
                st.info("Yeni dosya yok. Mevcut indeks yüklendi.")
            else:
                st.warning("Önce en az bir dosya yükleyin veya mevcut indeks bulunamadı.")

with col_b:
    vs_test = load_index()
    st.info("İndeks durumu: " + ("✅ Yüklü (RAG Etkin)" if vs_test else "❌ Yok (Normal Sohbet)"))

st.markdown("---")

st.subheader("💬 Soru Sor")
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

query = st.chat_input("Sorunu yaz…")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append(("user", query))
    
    history = format_history_for_prompt(st.session_state.messages)

    vs = load_index()
    llm = load_llm()

    with st.chat_message("assistant"):
        with st.spinner("Yanıt hazırlanıyor..."):
            try:
                if vs:
                    answer, used_docs = rag_answer(llm, vs, query, history, k=top_k)
                    st.markdown(answer)

                    if any("Kaynaklar:" in line for line in answer.split('\n')):
                        render_sources(used_docs)
                else:
                    answer = chat_answer(llm, query, history)
                    st.markdown(answer)
                
                st.session_state.messages.append(("assistant", answer))
            except Exception as e:
                st.error(f"Hata: {e}")
