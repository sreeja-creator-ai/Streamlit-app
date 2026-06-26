import streamlit as st
import tempfile
import os

from resume_parser import extract_resume_text
from skill_extractor import extract_skills
from rag import create_vector_store, generate_question
from evaluator import evaluate_answer


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="InterviewGPT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 InterviewGPT")
st.subheader("AI Interview Coach using RAG + Groq")

st.markdown("---")


# --------------------------------------------------
# Load Interview Knowledge Base
# --------------------------------------------------

PDF_PATH = "Interview Questions.pdf"

if not os.path.exists(PDF_PATH):
    st.error("Interview Questions.pdf not found in project folder.")
    st.stop()


@st.cache_resource
def load_vector_database():
    return create_vector_store(PDF_PATH)


vectordb = load_vector_database()


# --------------------------------------------------
# Upload Resume
# --------------------------------------------------

st.header("📄 Upload Resume")

resume = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)


if resume is not None:

    # Save uploaded resume temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(resume.read())

        resume_path = tmp.name

    # Extract Resume Text
    with st.spinner("Reading Resume..."):

        resume_text = extract_resume_text(
            resume_path
        )

    st.success("Resume Uploaded Successfully")

    st.subheader("Resume Preview")

    st.write(
        resume_text[:1000]
    )

    # Extract Skills
    skills = extract_skills(
        resume_text
    )

    st.subheader("Detected Skills")

    if skills:

        st.success(
            ", ".join(skills)
        )

    else:

        st.warning(
            "No skills detected."
        )

    st.markdown("---")

    # ----------------------------------------
    # Generate Personalized Question
    # ----------------------------------------

    if st.button(
        "Generate Personalized Interview Question"
    ):

        with st.spinner(
            "Generating Personalized Question..."
        ):

            question = generate_question(
                vectordb,
                skills
            )

            st.session_state.question = question

        st.success(
            "Question Generated Successfully"
        )


# --------------------------------------------------
# Display Interview Question
# --------------------------------------------------

if "question" in st.session_state:

    st.markdown("---")

    st.header(
        "🎤 Interview Question"
    )

    st.info(
        st.session_state.question
    )

    answer = st.text_area(
        "Enter Your Answer",
        height=200
    )

    if st.button(
        "Evaluate Answer"
    ):

        if answer.strip() == "":

            st.warning(
                "Please enter your answer."
            )

        else:

            with st.spinner(
                "Evaluating Answer..."
            ):

                feedback = evaluate_answer(
                    st.session_state.question,
                    answer
                )

            st.markdown("---")

            st.header(
                "📊 Interview Evaluation"
            )

            st.markdown(
                feedback
            )

st.markdown("---")

st.caption(
    "InterviewGPT | Resume Analysis | Personalized RAG | Groq"
)
