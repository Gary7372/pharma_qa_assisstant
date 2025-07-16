import streamlit as st
from graph_builder import build_graph
from langchain.schema import Document

st.set_page_config(page_title="💊 Pharma QA Assistant", layout="wide")
st.title("💊 Pharma QA Assistant")

query = st.text_input("Enter your medical/pharmaceutical question:")

if query:
    with st.spinner("Retrieving and analyzing documents..."):
        app = build_graph()
        result = app.invoke({"query": query})

        st.markdown("## ✅ Answer")
        st.write(result["answer"])

        st.markdown("## 📄 Retrieved Document Chunks")
        documents: list[Document] = result["documents"]
        for i, doc in enumerate(documents, 1):
            with st.expander(f"Chunk {i} — Page {doc.metadata.get('page', '?')}"):
                st.markdown(doc.page_content)
                st.caption(f"📎 Metadata: {doc.metadata}")
