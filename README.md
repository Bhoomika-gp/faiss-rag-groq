# faiss-rag-groq

## rag-chatbot
### University Info Chatbot | RAG + LLM for Harvard & Stanford CDS
higher Education decision assitant
RAG chatbot using FAISS, Sentence-Transformers and Groq models. Streamlit UI included.

## Run locally
1. create venv: `python -m venv env & source env/bin/activate` (Windows: `env\Scripts\activate`)
2. install: `pip install -r requirements.txt`
3. create `.env` with `GROQ_API_KEY` and `GROQ_MODEL`
4. run: `streamlit run app_streamlit.py`
