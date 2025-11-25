import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from src.data_loader import load_all_documents

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        llm_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  

        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"]["text"] for r in results if r["metadata"] and "text" in r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        
        prompt = f"""
    You are an academic comparison assistant. Use ONLY the provided context to answer.

    Query:
    {query}

    Context:
    {context}

    Instructions:
    - If comparing two institutions, structure your answer as a clear comparison table.
    - Cite each point by university name if mentioned in the context.
    - If any information is missing for one institution, explicitly say "Not available in context."
    - Do NOT hallucinate or add any outside knowledge.

    Answer:
    """
        response = self.llm.invoke([prompt])
        return response.content.strip()


# Example usage
# if __name__ == "__main__":
#     rag_search = RAGSearch()
#     query = "could you please give me the comparision table of stanford and harvard university annual expenses"
#     summary = rag_search.search_and_summarize(query, top_k=3)
#     print("Summary:", summary)