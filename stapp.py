import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

# Streamlit app
st.title("Gen AI App with Streamlit")

# Data Ingestion
st.header("Data Ingestion")
url = st.text_input("Enter the URL to scrape data from:", "https://docs.smith.langchain.com/tutorials/Administrators/manage_spend")
if st.button("Load Data"):
    loader = WebBaseLoader(url)
    docs = loader.load()
    st.write("Data loaded successfully!")

    # Data Processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstoredb = FAISS.from_documents(documents, embeddings)
    st.write("Data processed and stored in vector database!")

# Querying
query = st.text_input("Enter your query:", "LangSmith has two usage limits: total traces and extended")
if st.button("Search"):
    result = vectorstoredb.similarity_search(query)
    st.write("Query Result:", result[0].page_content)

    # Retrieval Chain
    llm = ChatOpenAI(model="gpt-4o")
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        """
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstoredb.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": query})
    st.write("LLM Response:", response['answer'])