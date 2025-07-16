#.venv\Scripts\python.exe -m streamlit run app.py
import os
#from dotenv import load_dotenv
#load_dotenv()
import streamlit as st

from graph_builder import build_graph

# Initialize graph once
graph = build_graph()

st.set_page_config(page_title="Pharma QA Assistant", layout="wide")
st.title("💊 Pharmaceutical QA Assistant")
st.write("Ask a question based on your parsed regulatory or clinical documents.")

# Text input
query = st.text_area("Enter your question:", height=100)

if st.button("Submit Question") and query.strip():
    with st.spinner("Analyzing documents..."):
        try:
            result = graph.invoke({"query": query})
            st.success("Answer generated successfully!")
            st.subheader("🧠 Answer:")
            st.markdown(result["answer"])

        except Exception as e:
            st.error(f"Error: {e}")
