
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import docx

# Initialize embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from different file types
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        reader = PdfReader(file)
        return "
".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        return "
".join([para.text for para in doc.paragraphs])
    else:
        return ""

# Function to categorize document type
def categorize(text):
    text = text.lower()
    if "troubleshooting" in text:
        return "Troubleshooting"
    elif "manual" in text:
        return "Manual"
    elif "log" in text or "survey" in text:
        return "Survey Log"
    else:
        return "Other"

# Streamlit UI
st.title("📄 AI-Powered Document Search Tool")

uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT)", accept_multiple_files=True)
query = st.text_input("Enter your search query:")

if uploaded_files and query:
    documents = []
    metadata = []

    for file in uploaded_files:
        text = extract_text(file)
        if text:
            documents.append(text)
            metadata.append({
                "filename": file.name,
                "type": categorize(text),
                "snippet": text[:300] + "..." if len(text) > 300 else text
            })

    # Generate embeddings
    doc_embeddings = model.encode(documents)
    query_embedding = model.encode([query])[0]

    # Compute similarity
    scores = cosine_similarity([query_embedding], doc_embeddings).flatten()
    results = sorted(zip(scores, metadata), key=lambda x: x[0], reverse=True)

    # Display results
    st.subheader("🔍 Search Results")
    for score, meta in results:
        st.markdown(f"**{meta['filename']}** ({meta['type']})")
        st.markdown(f"Similarity Score: `{score:.4f}`")
        st.write(meta['snippet'])
        st.markdown("---")
