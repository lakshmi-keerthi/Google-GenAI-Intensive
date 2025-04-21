import streamlit as st
import pandas as pd
import numpy as np
import json
import google.generativeai as genai

# ✅ This must be the first Streamlit command
st.set_page_config(page_title="Thera – Mental Health Therapist", layout="centered")

# 🔐 Load API key securely
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# 📦 Load vector store (stored embeddings + text)
@st.cache_resource
def load_vector_store():
    return pd.read_pickle("vector_store.pkl")

formatted_df = load_vector_store()

# 🔎 Embed the query using Gemini text-embedding-004
def embed_fn(text):
    response = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"])

# 🔎 Retrieve top-k most relevant historical examples
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

# 🧠 Generate Gemini-based therapist response
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

# 🎨 Chat UI

st.title("🧠 Thera – Your Mental Health Therapist")
st.markdown("Talk to Thera, your AI therapist. She listens with empathy and responds with care.")

# Initialize session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display full chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input as chat
if user_input := st.chat_input("What's on your mind?"):
    # Store & display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate Gemini response
    with st.spinner("Thera is thinking..."):
        raw_response = mental_health_rag_response(user_input)

    try:
        # Extract just the JSON
        start = raw_response.find("{")
        end = raw_response.rfind("}") + 1
        json_block = raw_response[start:end]
        parsed = json.loads(json_block)

        # Show assistant message
        response_text = parsed["response"]
        suggestions = parsed.get("suggestions", [])

        with st.chat_message("assistant"):
            st.markdown(response_text)
            if suggestions:
                st.markdown("🌱 **Suggestions:**")
                for s in suggestions:
                    st.markdown(f"- {s}")

        st.session_state.messages.append({"role": "assistant", "content": response_text})

    except Exception as e:
        fallback = "Hmm... I had trouble understanding that. Let’s try again."
        st.chat_message("assistant").markdown(fallback)
        st.session_state.messages.append({"role": "assistant", "content": fallback})
