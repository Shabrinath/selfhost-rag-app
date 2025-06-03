import streamlit as st
import os

# Suppress PyTorch + Streamlit file watching issues
#os.environ["TORCH_DISABLE_USER_OPS_WARNINGS"] = "1"
#os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# Paths
DATA_DIR = "./data"
DB_DIR = "./db"

@st.cache_resource
def load_vectorstore():
    """Loads or builds the Chroma vector store."""
    if not os.path.exists(DB_DIR) or not os.listdir(DB_DIR):
        st.info("Indexing documents...")

        loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", show_progress=True)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(splits, embedding=embeddings, persist_directory=DB_DIR)
        vectorstore.persist()
    else:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    return vectorstore

def main():
    st.set_page_config(page_title="RAG Assistant", layout="wide")
    st.title("📄 Internal Q&A with RAG (Docs + Ollama)")

    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    llm = OllamaLLM(model="llama3")
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    query = st.text_input("Ask a question about your documents:", placeholder="e.g. What is our refund policy?")

    if query:
        with st.spinner("Searching your documents..."):
            result = qa_chain.invoke({"query": query})
            st.success(result["result"])

            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(result["source_documents"]):
                    st.markdown(f"**Document {i+1}**: `{doc.metadata.get('source')}`")
                    st.write(doc.page_content[:500] + "...")


if __name__ == "__main__":
    main()
