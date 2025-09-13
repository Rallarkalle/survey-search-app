
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import tempfile
import docx2txt
import fitz  # PyMuPDF

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Session state for chat history and document chunks
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'document_chunks' not in st.session_state:
    st.session_state.document_chunks = []

# Function to extract text from uploaded files
def extract_text(file):
    if file.type == "application/pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = "
".join([page.get_text() for page in doc])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file.read())
            text = docx2txt.process(tmp.name)
    elif file.type == "text/plain":
        text = file.read().decode("utf-8")
    else:
        text = ""
    return text

# Sidebar for uploading documents
st.sidebar.header("Upload Survey Documents")
uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Process uploaded files
if uploaded_files:
    all_text = ""
    for file in uploaded_files:
        text = extract_text(file)
        all_text += text + "
"

    # Split text into chunks and embed
    chunks = [chunk.strip() for chunk in all_text.split("
") if len(chunk.strip()) > 20]
    embeddings = model.encode(chunks, convert_to_tensor=True)
    st.session_state.document_chunks = list(zip(chunks, embeddings))

# Chatbot interface
st.title("Survey Troubleshooting Assistant")
st.write("Ask any troubleshooting question related to your survey work. The assistant will search your uploaded documents and provide intelligent suggestions.")

user_input = st.text_input("You:", "", key="user_input")

if user_input:
    # Embed user query
    query_embedding = model.encode(user_input, convert_to_tensor=True)

    # Find most relevant chunk
    if st.session_state.document_chunks:
        scores = [util.pytorch_cos_sim(query_embedding, emb)[0][0].item() for _, emb in st.session_state.document_chunks]
        best_idx = scores.index(max(scores))
        best_chunk = st.session_state.document_chunks[best_idx][0]

        # Generate response (simple version)
        response = f"Based on the documentation, here's a suggestion:

{best_chunk}"
    else:
        response = "Please upload relevant documents first."

    # Update chat history
    st.session_state.chat_history.append((user_input, response))

# Display chat history
st.markdown("---")
for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {user_msg}")
    st.markdown(f"**Assistant:** {bot_msg}")
    st.markdown("---")
