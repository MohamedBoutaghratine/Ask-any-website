#pip install langchain langchain-openai

import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



# Load embeddings from HuggingFace
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize or load a Chroma vector store



def get_vectorstore_from_url(url):
    
    persist_directory1 = "./chroma_db"

    #get the text from website
    loader = WebBaseLoader(url)
    documents = loader.load()
    #doc = documents[1].page_content
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=50
    )
    document_chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(document_chunks, embedding_model, persist_directory=persist_directory1)

    return vector_store

#define LLM
llm = ChatGroq(
    temperature = 0.3 ,
    groq_api_key ='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',
    model_name ='llama-3.1-70b-versatile' 

)
#define RAG

def get_context_retriever_chain(vector_store):

    retriever = vector_store.as_retriever(search_type='similarity',search_kwargs={"k":5})
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name='chat_history'),
        ('user',"{input}"),
        ('user'," Given the above conversation, generate a search query to look up in order to get information relevant to the conversation to answer to the queries or question of the users by searching in the url given and always keep the context in the chat_history in your mind.")

    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt) 

    return retriever_chain


def get_conversational_rag_chain (retriever_chain):
    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversational_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input":user_input
        })

    return response['answer']


#app configuration
st.set_page_config(page_title="Ask website", page_icon="🤖")
st.title("Ask Any Website")



#sidebar
with st.sidebar :
    st.header("WEBSITE")
    website_url = st.text_input('Website URL')
if website_url is None or website_url == "" :
    st.write("Please enter a Website URL")
else :
    # session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello,I'm a bot, how can I help you ?"),
    ]
    if "vector_store" not in st.session_state or st.session_state.website_url != website_url:
        # Re-initialize vector store for the new website
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
        # Store the new URL in session state to track changes
        st.session_state.website_url = website_url



    #user input
    user_query = st.chat_input("Write your question...")
    if user_query is not None and  user_query != "" :
        response = get_response(user_query)

        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

        #retrieved_doc = retriever_chain.invoke({
        #   "chat_history": st.session_state.chat_history,
        #    "input": user_query
        #})
        #st.write(retrieved_doc)

       
    #conversation
    for message in st.session_state.chat_history :
        if isinstance (message,AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
