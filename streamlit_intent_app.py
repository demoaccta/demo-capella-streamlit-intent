# Import Libraries
import streamlit as st
import openai
from couchbase.cluster import Cluster
from couchbase.auth import PasswordAuthenticator
from couchbase.options import ClusterOptions, QueryOptions
from couchbase.vector_search import VectorQuery, VectorSearch
from couchbase.search import SearchRequest
from datetime import timedelta

import os

# Load env variables
CB_CONN_STR = st.secrets["COUCHBASE_CONN_STR"]
CB_USERNAME = st.secrets["COUCHBASE_USERNAME"]
CB_PASSWORD = st.secrets["COUCHBASE_PASSWORD"]
CB_BUCKET = st.secrets["COUCHBASE_BUCKET"]
CB_SCOPE = st.secrets["COUCHBASE_SCOPE"]
CB_COLLECTION = st.secrets["COUCHBASE_COLLECTION"]

openai.api_key = st.secrets["OPENAI_API_KEY"]

# Connect to Couchbase

@st.cache_resource
def get_cb_cluster():
    auth = PasswordAuthenticator(CB_USERNAME, CB_PASSWORD)
    options = ClusterOptions(
      authenticator=auth,
      kv_timeout=timedelta(seconds=20),
      config_idle_redial_timeout=timedelta(seconds=20)
    )
    cluster = Cluster(CB_CONN_STR, options)
    cluster.wait_until_ready(timedelta(seconds=10))
    return cluster

cluster = get_cb_cluster()
bucket = cluster.bucket(CB_BUCKET)
scope = bucket.scope(CB_SCOPE)

# Generate Query Embedding

def generate_embedding(text: str):
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# Cache Query Embedding
@st.cache_data(show_spinner=False)
def cached_embedding(text):
    return generate_embedding(text)

# Vector Search
def vector_search(scope, query_embedding, top_k=5):
    sql = """
    SELECT bp.* 
    FROM `rag_capella`.`standard_reports`.`balance_position` AS bp
    WHERE bp.embedding IS NOT NULL
    ORDER BY VECTOR_DISTANCE(bp.embedding, $embedding, "dot")
    LIMIT $top_k
    """
    print(sql)
    result = scope.query(
        sql,
        QueryOptions(
            named_parameters={
                "embedding": query_embedding,
                "top_k": top_k
            }
        )
    )

    docs = []
    for row in result:
        docs.append(str(row))

    return docs
    
# RAG Prompt Construction

def build_prompt(question, contexts):
    context_text = "\n\n".join(contexts)
    prompt = f"""
You are an assistant answering based only on the context below.

Context:
{context_text}

Question:
{question}

Answer clearly and concisely.
"""
    return prompt

# LLM completion

def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

# Streamlit UI layer

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Welcome to FOurSee CONVui", layout="centered")
st.title("Welcome to FOurSee CONVui")

# Render Existing Chat History
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(chat["question"])

    with st.chat_message("assistant"):
        st.markdown(chat["answer"])
    
# Get User Input (Query)
user_query = st.chat_input("Ask a question", key="user_query")

# Process New Query 
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)
    with st.spinner("Thinking..."):
        query_embedding = cached_embedding(user_query)
        retrieved_docs = vector_search(scope, query_embedding,top_k=1)
        prompt = build_prompt(user_query, retrieved_docs)
        answer = generate_answer(prompt)
    with st.chat_message("assistant"):
        st.markdown(answer)
    #st.subheader("Answer")
    #st.write(answer)
    #st.write(retrieved_docs)

    # Append to Session State
    st.session_state.chat_history.append({
        "question": user_query,
        "answer": answer
    })

    #st.session_state.user_query = ""
