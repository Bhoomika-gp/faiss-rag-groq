import streamlit as st
from src.search import RAGSearch

# Page config
st.set_page_config(page_title="higher Education decision assitant")

# Title
st.title("University FAQ Chatbot")
st.write("Your personal assistant for university info—ask anything from admissions to fees!")

# Initialize RAGSearch (only once)
@st.cache_resource(show_spinner=False)
def init_rag():
    return RAGSearch()

rag_search = init_rag()

# Input query
query = st.text_input("Enter your question:")

if st.button("Get Summary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching and summarizing..."):
            try:
                summary = rag_search.search_and_summarize(query, top_k=3)
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"Error: {e}")
