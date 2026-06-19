# app.py

import streamlit as st
import tempfile

from resume_parser import extract_resume_text
from skill_extractor import extract_skills
from rag import create_vector_store, generate_question
from evaluator import evaluate_answer

st.set_page_config(
    page_title="InterviewGPT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 InterviewGPT")
st.subheader("AI Interview Coach using RAG + Groq")

st.markdown("---")

# -----------------------------
# Resume Upload
# -----------------------------

st.header("📄 Upload Resume")

resume = st.file_uploader(
    "Upload Resume PDF",
    type=["pdf"]
)

if resume is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(resume.read())
        resume_path = tmp.name

    resume_text = extract_resume_text(
        resume_path
    )

    st.success("Resume Uploaded Successfully")

    st.subheader("Resume Preview")

    st.write(
        resume_text[:1000]
    )

    skills = extract_skills(
        resume_text
    )

    st.subheader("Detected Skills")

    if skills:
        st.write(skills)
    else:
        st.warning(
            "No skills detected."
        )

st.markdown("---")

# -----------------------------
# Interview PDF Upload
# -----------------------------

st.header("📚 Upload Interview Questions PDF")

interview_pdf = st.file_uploader(
    "Upload Interview Questions PDF",
    type=["pdf"]
)

if interview_pdf is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(
            interview_pdf.read()
        )

        pdf_path = tmp.name

    if st.button(
        "Generate Interview Question"
    ):

        with st.spinner(
            "Generating Question..."
        ):

            vectordb = create_vector_store(
                pdf_path
            )

            question = generate_question(
                vectordb
            )

            st.session_state.question = (
                question
            )

        st.success(
            "Question Generated Successfully"
        )

# -----------------------------
# Question Display
# -----------------------------

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
                "Please enter an answer."
            )

        else:

            with st.spinner(
                "Evaluating..."
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
    "InterviewGPT | Resume Analysis | RAG | Groq"
)