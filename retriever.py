from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain.schema import Document


def get_retriever():
    # Initialize SentenceTransformer embeddings
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    class SentenceTransformerWrapper:
        def embed_documents(self, texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    embeddings = SentenceTransformerWrapper()

    # Load FAISS index with embeddings object (Note: FAISS expects LangChain Embeddings interface;
    # since SentenceTransformerWrapper does not fully implement it, loading index might require adjustment)
    # If FAISS.load_local requires LangChain embeddings, consider using langchain.embeddings.SentenceTransformerEmbeddings
    from langchain.embeddings import SentenceTransformerEmbeddings
    embeddings_lc = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = FAISS.load_local(
        "faiss_db",
        embeddings_lc,
        allow_dangerous_deserialization=True
    )

    # Return retriever that yields full Document objects (with metadata)
    return db.as_retriever(search_type="similarity", k=10)
