from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import random


# ---------------------------------------------------
# Create Chroma Vector Database
# ---------------------------------------------------

def create_vector_store(pdf_file):

    loader = PyPDFLoader(pdf_file)

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


# ---------------------------------------------------
# Generate Personalized Interview Question
# ---------------------------------------------------

def generate_question(vectordb, skills):

    # -----------------------------
    # Build Search Query
    # -----------------------------

    if skills and len(skills) > 0:

        query = f"""
        Interview questions for a candidate skilled in:
        {", ".join(skills)}

        Focus only on these technologies.
        """

    else:

        query = "Technical interview questions"

    # -----------------------------
    # Retrieve Relevant Chunks
    # -----------------------------

    docs = vectordb.similarity_search(
        query=query,
        k=10
    )

    questions = []

    # -----------------------------
    # Extract Questions
    # -----------------------------

    for doc in docs:

        text = doc.page_content

        parts = text.split("?")

        for part in parts:

            part = part.strip()

            if len(part) > 10:

                question = part + "?"

                questions.append(question)

    # -----------------------------
    # Remove Duplicates
    # -----------------------------

    questions = list(dict.fromkeys(questions))

    # -----------------------------
    # Return Question
    # -----------------------------

    if len(questions) == 0:

        return (
            "No interview question found "
            "for the detected skills."
        )

    return random.choice(questions)


# ---------------------------------------------------
# Test
# ---------------------------------------------------

if __name__ == "__main__":

    pdf = "Interview Questions.pdf"

    vectordb = create_vector_store(pdf)

    skills = [
        "Python",
        "Machine Learning",
        "SQL",
        "Pandas"
    ]

    for i in range(5):

        print()

        print("Question", i + 1)

        print("-" * 60)

        print(
            generate_question(
                vectordb,
                skills
            )
        )

        print("-" * 60)
