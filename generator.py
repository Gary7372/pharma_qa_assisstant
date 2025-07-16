import os
#from dotenv import load_dotenv
#load_dotenv()

from langchain_groq import ChatGroq
from langchain.schema import Document, SystemMessage, HumanMessage

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    reasoning_format="parsed"
)

def generate_answer(query: str, documents: list[Document]) -> str:
    # Combine document content with optional metadata (e.g., page number)
    context = "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')}] {doc.page_content.strip()}" for doc in documents
    )

    messages = [
        SystemMessage(
            content="You are a pharmaceutical research assistant. Analyze the provided documents and answer the question with precise information from the documents."
        ),
        HumanMessage(
            content=f"""Question: {query}

Relevant Documents:
{context}

Instructions:
1. Answer ONLY using information from the provided documents.
2. Be precise with numbers, dosages, and medical terms.
3. Include page references when possible."""
        )
    ]

    response = llm(messages)
    return response.content if hasattr(response, "content") else str(response)

