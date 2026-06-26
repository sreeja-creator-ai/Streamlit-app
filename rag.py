from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import random


# -------------------------------------------------------
# Create Vector Store
# -------------------------------------------------------

def create_vector_store(pdf_file):

    loader = PyPDFLoader(pdf_file)

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


# -------------------------------------------------------
# Generate Personalized Interview Question
# -------------------------------------------------------

def generate_question(vectordb, skills):

    # If no skills are detected, use a generic query
    if skills:

        query = (
            "Interview questions related to "
            + ", ".join(skills)
        )

    else:

        query = "Technical interview questions"

    # Retrieve relevant documents
    docs = vectordb.similarity_search(
        query=query,
        k=10
    )

    questions = []

    # Extract questions from retrieved documents
    for doc in docs:

        text = doc.page_content

        lines = text.split("?")

        for line in lines:

            line = line.strip()

            if len(line) > 5:

                question = line + "?"

                questions.append(question)

    # Remove duplicate questions
    questions = list(dict.fromkeys(questions))

    # Return a relevant question
    if questions:

        return random.choice(questions)

    return "No relevant interview question found."


# -------------------------------------------------------
# Testing
# -------------------------------------------------------

if __name__ == "__main__":

    pdf_file = r"Interview Questions.pdf"

    vectordb = create_vector_store(pdf_file)

    skills = [
        "Python",
        "SQL",
        "Machine Learning"
    ]

    print(generate_question(vectordb, skills))
