import streamlit as st
import tempfile

from resume_parser import extract_resume_text
from skill_extractor import extract_skills
from rag import create_vector_store, generate_question
from evaluator import evaluate_answer
import os


# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(
    page_title="InterviewGPT",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 InterviewGPT")
st.subheader("AI Interview Coach using Resume + RAG + Groq")

st.markdown("---")


# --------------------------------------------------
# Load Interview Question Knowledge Base
# --------------------------------------------------

PDF_PATH = "Interview_Questions.pdf"

# Create vector database only once
@st.cache_resource
def load_vector_db():
    return create_vector_store(PDF_PATH)

vectordb = load_vector_db()


# --------------------------------------------------
# Resume Upload
# --------------------------------------------------

st.header("📄 Upload Resume")

resume = st.file_uploader(
    "Upload your Resume (PDF)",
    type=["pdf"]
)


if resume is not None:

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(resume.read())
        resume_path = tmp.name


    # -------------------------
    # Resume Text
    # -------------------------

    resume_text = extract_resume_text(
        resume_path
    )

    st.success("Resume Uploaded Successfully ✅")

    with st.expander("Resume Preview"):

        st.write(
            resume_text[:1500]
        )


    # -------------------------
    # Skills
    # -------------------------

    skills = extract_skills(
        resume_text
    )

    st.subheader("🛠 Extracted Skills")

    if skills:

        st.success(", ".join(skills))

    else:

        st.warning(
            "No skills detected."
        )



    st.markdown("---")


    # --------------------------------------------------
    # Personalized Question
    # --------------------------------------------------

    if st.button(
        "Generate Personalized Interview Question"
    ):

        with st.spinner(
            "Generating Question..."
        ):

            question = generate_question(
                vectordb,
                skills
            )

            st.session_state.question = question



# --------------------------------------------------
# Display Question
# --------------------------------------------------

if "question" in st.session_state:

    st.header("🎤 Personalized Interview Question")

    st.info(
        st.session_state.question
    )


    answer = st.text_area(
        "Enter Your Answer",
        height=220
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

            st.header("📊 Interview Feedback")

            st.markdown(feedback)



st.markdown("---")

st.caption(
    "InterviewGPT | Resume Parsing | Skill Extraction | RAG | Groq"
)
