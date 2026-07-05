import streamlit as st
import plotly.graph_objects as go

from rag.retrive import retrieve_similar_cases
from llm.groq_llm import analyze

st.set_page_config(
    page_title="MindMatrix",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Risk Meter Function
# ---------------------------

def show_risk_meter(risk_level):

    risk_map = {
        "low": 25,
        "moderate": 50,
        "high": 75,
        "critical": 100
    }

    value = risk_map.get(
        risk_level.strip().lower(),
        50
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": "Mental Health Risk Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "red"},
            "steps": [
                {"range": [0, 25], "color": "lightgreen"},
                {"range": [25, 50], "color": "yellow"},
                {"range": [50, 75], "color": "orange"},
                {"range": [75, 100], "color": "red"}
            ]
        }
    ))

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ---------------------------
# Header
# ---------------------------

st.title("🧠 MindMatrix")
st.markdown(
    "### A RAG + LLM Based System for Multi-Class Mental Health Risk Assessment"
)

st.markdown("---")

# ---------------------------
# Horizontal Navigation
# ---------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "📝 Assessment",
        "📊 Dataset Information",
        "ℹ️ About Project"
    ]
)

# =====================================================
# TAB 1 : ASSESSMENT
# =====================================================

with tab1:

    st.subheader("Mental Health Assessment")

    user_text = st.text_area(
        "Describe how you have been feeling:",
        height=200,
        placeholder="Example: I feel anxious all day and cannot sleep..."
    )

if st.button("Analyze"):

    if not user_text.strip():
        st.warning("Please enter some text.")
        st.stop()

    with st.spinner("Retrieving similar cases..."):

        results = retrieve_similar_cases(user_text)

        retrieved_context = ""

        for doc, meta in zip(
            results["documents"][0],
            results["metadatas"][0]
        ):
            retrieved_context += (
                f"Label: {meta['label']}\n"
                f"Text: {doc}\n\n"
            )

    st.subheader("Retrieved Evidence")
    st.info(retrieved_context)

    with st.spinner("Generating assessment..."):

        output = analyze(
            user_text,
            retrieved_context
        )
        st.write(output)
    # --------------------------------
    # Extract Values from LLM Output
    # --------------------------------

    condition = "Unknown"
    confidence = "N/A"
    risk_level = "Moderate"

    for line in output.split("\n"):

        if line.startswith("Possible Condition:"):
            condition = line.replace(
                "Possible Condition:",
                ""
            ).strip()

        elif line.startswith("Confidence Score:"):
            confidence = line.replace(
                "Confidence Score:",
                ""
            ).strip()

        elif line.startswith("Risk Level:"):
            risk_level = line.replace(
                "Risk Level:",
                ""
            ).strip()

    # --------------------------------
    # Dashboard Cards
    # --------------------------------

    st.subheader("Assessment Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Condition",
            value=condition
        )

    with col2:
        st.metric(
            label="Confidence",
            value=confidence
        )

    with col3:
        st.metric(
            label="Risk Level",
            value=risk_level
        )

    # --------------------------------
    # Risk Meter
    # --------------------------------

    st.subheader("Risk Assessment")
    st.write("Risk Level Found:", risk_level)
    show_risk_meter(risk_level)

    # --------------------------------
    # Full Report
    # --------------------------------

    st.subheader("Assessment Report")

    st.markdown(output)

# =====================================================
# TAB 2 : DATASET
# =====================================================

with tab2:

    st.subheader("Dataset Information")

    st.markdown("""
### Dataset Source

Reddit Mental Health Dataset

### Mental Health Categories

- Depression
- Anxiety
- ADHD
- Bipolar Disorder
- PTSD
- Control

### Purpose

The dataset is used to retrieve semantically
similar mental health narratives using
Sentence Transformers and ChromaDB.

The retrieved evidence is then supplied
to a Large Language Model (LLaMA 3.1)
through a Retrieval-Augmented Generation
(RAG) pipeline.
""")

# =====================================================
# TAB 3 : ABOUT
# =====================================================

with tab3:

    st.subheader("About MindMatrix")

    st.markdown("""
### Project Title

MindMatrix: A RAG + LLM Based System for
Multi-Class Mental Health Disorder Identification
from Reddit User Histories

### Technologies Used

- Python
- Streamlit
- ChromaDB
- Sentence Transformers
- Groq API
- LLaMA 3.1
- Retrieval Augmented Generation (RAG)

### Workflow

User Input
→ Embedding Generation
→ ChromaDB Retrieval
→ Similar Case Retrieval
→ LLaMA Analysis
→ Risk Assessment
→ Recommendations

### Developed For

M.Tech CSE Project
""")
# import streamlit as st
# from dotenv import load_dotenv
# import os

# from rag.retrive import retrieve_similar_cases
# from llm.groq_llm import analyze

# load_dotenv()

# st.set_page_config(
#     page_title="MindMatrix",
#     page_icon="🧠",
#     layout="wide"
# )

# st.title("🧠 MindMatrix")
# st.subheader("Mental Health Risk Assessment System")

# st.markdown("---")

# user_text = st.text_area(
#     "Describe how you have been feeling:",
#     height=200,
#     placeholder="Example: I feel anxious all day and cannot sleep..."
# )

# if st.button("Analyze"):

#     if not user_text.strip():
#         st.warning("Please enter some text.")
#         st.stop()

#     with st.spinner("Retrieving similar cases..."):

#         results = retrieve_similar_cases(user_text)

#         retrieved_context = ""

#         for doc, meta in zip(
#             results["documents"][0],
#             results["metadatas"][0]
#         ):
#             retrieved_context += (
#                 f"Label: {meta['label']}\n"
#                 f"Text: {doc}\n\n"
#             )

#     st.subheader("Retrieved Evidence")

#     st.text(retrieved_context)

#     with st.spinner("Generating assessment..."):

#         output = analyze(
#             user_text,
#             retrieved_context
#         )

#     st.subheader("Assessment")

#     st.markdown(output)