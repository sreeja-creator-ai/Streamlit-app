from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import random


# -------------------------------------------------------
# Create Vector Store
# -------------------------------------------------------

def create_vector_store(pdf_file):

    # Load PDF
    loader = PyPDFLoader(pdf_file)

    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create Chroma Vector DB
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


# -------------------------------------------------------
# Generate Personalized Question
# -------------------------------------------------------

def generate_question(vectordb, skills):

    # If no skills are detected
    if not skills:

        search_query = "Technical interview questions"

    else:

        # Convert skills list to query
        search_query = (
            "Interview questions on "
            + ", ".join(skills)
        )

    # Retrieve relevant chunks
    docs = vectordb.similarity_search(
        search_query,
        k=10
    )

    questions = []

    # Extract questions from retrieved chunks
    for doc in docs:

        text = doc.page_content

        # Split using '?'
        parts = text.split("?")

        for part in parts:

            part = part.strip()

            if len(part) > 10:

                question = part + "?"

                questions.append(question)

    # Remove duplicates
    questions = list(dict.fromkeys(questions))

    # Return one relevant question
    if len(questions) > 0:

        return random.choice(questions)

    return "No relevant interview question found."


# -------------------------------------------------------
# Test
# -------------------------------------------------------

if __name__ == "__main__":

    pdf_file = "Interview Questions.pdf"

    vectordb = create_vector_store(pdf_file)

    skills = [
        "Python",
        "SQL",
        "Machine Learning"
    ]

    for i in range(5):

        print("-" * 60)

        print(
            generate_question(
                vectordb,
                skills
            )
        )

        print("-" * 60)
