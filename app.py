import streamlit as st
import base64
import time
from src.vectors import EmbeddingsManager
from src.chatbot import ChatbotManager

# Load and encode logo as base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("assets/logo.png")

# PDF display
def display_pdf(file):
    try:
        base64_pdf = base64.b64encode(file.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="400" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Unable to display PDF: {e}")

# Session state
if 'temp_pdf_path' not in st.session_state:
    st.session_state['temp_pdf_path'] = None
if 'chatbot_manager' not in st.session_state:
    st.session_state['chatbot_manager'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'embedding_ready' not in st.session_state:
    st.session_state['embedding_ready'] = False

# App config
st.set_page_config(page_title="DocIQ", layout="wide")

# Sidebar with styled logo and text
with st.sidebar:
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] {{
            background-color: #fdf6f6;
        }}
        </style>

        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 50vh;
            text-align: center;
        ">
            <img src="data:image/png;base64,{img_base64}" style="width: 120px; height: 120px; border-radius: 50%; object-fit: cover;" />
            <h2 style="margin: 0;">DocIQ</h2>
            <p style="margin-top: 5px; font-size: 16px;">
                Your intelligent assistant for documents
            </p>
            <hr style="width: 80%; margin: 20px 0;">
            <p style="font-size: 13px; color: grey;">© 2025 DocIQ by ENG.Asmaa</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main title
st.title("Chat with your Document")

# Upload inside chat area
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

# Auto process uploaded file
if uploaded_file and not st.session_state['embedding_ready']:
    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.session_state['temp_pdf_path'] = temp_path

    with st.spinner("Generating embeddings..."):
        try:
            embeddings_manager = EmbeddingsManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                qdrant_url="http://localhost:6333",
                collection_name="vector_db"
            )
            result = embeddings_manager.create_embeddings(temp_path)
            st.session_state['embedding_ready'] = True

            st.session_state['chatbot_manager'] = ChatbotManager(
                model_name="BAAI/bge-small-en",
                device="cpu",
                encode_kwargs={"normalize_embeddings": True},
                llm_model="llama3",
                llm_temperature=0.7,
                qdrant_url="http://localhost:6333",
                collection_name="vector_db"
            )
            st.success("PDF processed successfully.")
        except Exception as e:
            st.error(f"Error: {e}")

# Display PDF after it's processed
if st.session_state['temp_pdf_path']:
    with open(st.session_state['temp_pdf_path'], "rb") as f:
        display_pdf(f)

# Display chat history
if st.session_state['messages']:
    for msg in st.session_state['messages']:
        st.chat_message(msg['role']).markdown(msg['content'])

# Chat input
user_input = st.chat_input("Ask something about the document...")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state['messages'].append({"role": "user", "content": user_input})

    if not st.session_state['chatbot_manager'] or not st.session_state['embedding_ready']:
        response = "Please upload a PDF before starting the chat."
    else:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state['chatbot_manager'].get_response(user_input)
                time.sleep(1)
            except Exception as e:
                response = f"An error occurred: {e}"

    st.chat_message("assistant").markdown(response)
    st.session_state['messages'].append({"role": "assistant", "content": response})
