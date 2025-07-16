from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from typing import TypedDict, List
from retriever import get_retriever
from generator import generate_answer
from langchain.schema import Document

retriever = get_retriever()

class GraphState(TypedDict):
    query: str
    documents: List[Document]
    answer: str


def retrieve_docs(state: GraphState) -> GraphState:
    docs = retriever.invoke(state["query"])  # docs is a list of Document objects

    # Print retrieved chunks and metadata
    print(f"\n=== Retrieved {len(docs)} documents ===")
    for i, doc in enumerate(docs, 1):
        print(f"--- Document {i} ---")
        print(f"Content:\n{doc.page_content[:500]}")  # Print first 500 chars to keep it manageable
        print(f"Metadata: {doc.metadata}\n")

    return {"query": state["query"], "documents": docs}

def generate_node(state: GraphState) -> GraphState:
    answer = generate_answer(state["query"], state["documents"])
    return {**state, "answer": answer}

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("retriever", RunnableLambda(retrieve_docs))
    graph.add_node("generate", RunnableLambda(generate_node))
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
