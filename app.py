import streamlit as st
import pandas as pd
import numpy as np
import json
import google.generativeai as genai

# ✅ Must be first Streamlit command
st.set_page_config(page_title="Thera – Mental Health Therapist", layout="centered")

# --- Configure Gemini API ---
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# --- Load vector store ---
@st.cache_resource
def load_vector_store():
    return pd.read_pickle("vector_store.pkl")

formatted_df = load_vector_store()

# --- Embedding function ---
def embed_fn(text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

# --- Vector search using cosine similarity ---
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

# --- Gemini-powered structured response ---
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

Respond strictly in this JSON format:
{{
  "response": "Your therapist's reply using natural language and warmth",
  "suggestions": ["A list of simple mental wellness tips like sleep well, talk to someone, journal, etc."]
}}
"""

    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0].text

# --- Streamlit Chat UI ---
st.title("🧠 Thera – Your Mental Health Therapist")
st.markdown("Talk to Thera, your GenAI-powered therapist who responds with kindness and empathy.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("What's on your mind?")

if user_input:
    # Show user input
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Gemini responds
    with st.spinner("Thera is thinking..."):
        raw_response = mental_health_rag_response(user_input)

        try:
            start = raw_response.find("{")
            end = raw_response.rfind("}") + 1
            json_block = raw_response[start:end]
            parsed = json.loads(json_block)

            thera_reply = parsed["response"]
            suggestions = parsed.get("suggestions", [])

            # Show assistant reply
            with st.chat_message("assistant"):
                st.markdown(thera_reply)
                if suggestions:
                    st.markdown("🌱 **Suggestions:**")
                    for item in suggestions:
                        st.markdown(f"- {item}")

            st.session_state.messages.append({"role": "assistant", "content": thera_reply})

        except Exception as e:
            fallback_reply = "Hmm... I’m having trouble understanding that. Can we try again?"
            st.chat_message("assistant").markdown(fallback_reply)
            st.session_state.messages.append({"role": "assistant", "content": fallback_reply})
