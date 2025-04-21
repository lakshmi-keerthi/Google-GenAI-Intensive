import streamlit as st
import pandas as pd
import numpy as np
import json
import google.generativeai as genai

# --- Configure Gemini with secure key from Streamlit secrets ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Load vector store (pre-embedded dataframe) ---
@st.cache_resource
def load_vector_store():
    return pd.read_pickle("vector_store.pkl")

formatted_df = load_vector_store()

# --- Embedding function using text-embedding-004 ---
def embed_fn(text):
    model = genai.GenerativeModel("models/text-embedding-004")
    response = model.embed_content(
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

# --- Vector search: retrieve top-k similar past examples ---
def retrieve_similar_responses(query, df, top_k=3):
    query_vec = embed_fn(query)
    similarities = []

    for _, row in df.iterrows():
        score = np.dot(query_vec, np.array(row["embedding"]))
        similarities.append({
            "user": row["user"],
            "therapist": row["therapist"],
            "score": score
        })

    return sorted(similarities, key=lambda x: x["score"], reverse=True)[:top_k]

# --- Main RAG-based response generator using Gemini ---
def mental_health_rag_response(query):
    top_matches = retrieve_similar_responses(query, formatted_df)

    context = "\n".join([
        f"User: {item['user']}\nTherapist: {item['therapist']}"
        for item in top_matches
    ])

    prompt = f"""
You are Thera, a licensed and kind mental health therapist. Always respond gently and constructively. Respond with empathy.

- Use natural language with contractions (I'm, you’re, let’s)
- Occasionally include thoughtful pauses like: "Hmm, let’s think about that..."
- Talk less, listen more.
- Do not suggest or mention any medications.
- If the topic is out of scope, politely state that you cannot answer it.

Use the following past examples for context:

{context}

User: {query}

Respond strictly in the following JSON format:
{{
  "response": "Your therapist's reply using natural language and warmth",
  "suggestions": ["A list of simple mental wellness tips like sleep well, talk to someone, journal, etc."]
}}
"""

    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.candidates[0].content.parts[0].text


# --- Streamlit App UI ---
st.set_page_config(page_title="Thera – Mental Health Therapist", layout="centered")
st.title("🧠 Thera – Your Mental Health Therapist")
st.markdown("Ask Thera how you're feeling and get supportive suggestions.")

query = st.text_area("💬 What's on your mind?", height=150)

if st.button("Ask Thera"):
    if query.strip():
        with st.spinner("Thera is thinking..."):
            raw_response = mental_health_rag_response(query)

        try:
            parsed = json.loads(raw_response)

            st.subheader("🧠 Therapist's Response")
            st.markdown(parsed["response"])

            st.subheader("🌱 Suggestions")
            for s in parsed["suggestions"]:
                st.markdown(f"- {s}")

        except Exception as e:
            st.error("⚠️ Couldn't parse Gemini's response.")
            st.code(raw_response)

    else:
        st.info("Please type your question above.")
