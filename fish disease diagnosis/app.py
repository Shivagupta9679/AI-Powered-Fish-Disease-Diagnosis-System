import streamlit as st
from PIL import Image
from huggingface_hub import hf_hub_download
from predict import predict_image
from tensorflow.keras.models import load_model
from rag import retrieve_information
from llm import generate_diagnosis



# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Fish Disease Detection",
    page_icon="🐟",
    layout="wide"
)

@st.cache_resource
def load_my_model():

    model_path = hf_hub_download(
        repo_id="SHIVA9679/VGG16",
        filename="VGG16_fish_model_fixed.h5"
    )

    model = load_model(model_path, compile=False)

    return model
# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">🐟 Fish Disease Diagnosis System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-assisted fish disease prediction using VGG16 + RAG + LLM</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("About")

    st.write(
        """
        Upload a fish image to obtain an AI-assisted
        health classification.

        The system combines:

        • VGG16 image classification  
        • RAG-based knowledge retrieval  
        • LLM-generated diagnosis report
        """
    )

    st.divider()

    st.warning(
        "This system provides AI-assisted information "
        "and should not replace professional diagnosis."
    )


# -----------------------------
# Image Upload
# -----------------------------
st.subheader("📷 Upload Fish Image")

uploaded_file = st.file_uploader(
    "Choose a fish image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Symptoms Input
# -----------------------------
st.subheader("📝 Fish Symptoms")

symptoms = st.text_area(
    "Enter observed symptoms (optional)",
    placeholder=(
        "Example: white patches, skin lesions, abnormal swimming, "
        "loss of appetite..."
    )
)


# -----------------------------
# Diagnosis Button
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image,
            caption="Uploaded Fish Image",
            use_container_width=True
        )

    with col2:

        st.info("Image uploaded successfully.")

        if st.button(
            "🔍 Diagnose Fish",
            type="primary",
            use_container_width=True
        ):

            # -----------------------------
            # VGG16 Prediction
            # -----------------------------
            with st.spinner("Analyzing fish image..."):

                predicted_class, confidence = predict_image(image)

            st.success("Image analysis completed.")

            # -----------------------------
            # Display Prediction
            # -----------------------------
            st.markdown("### 🧠 AI Prediction")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric(
                    "Predicted Class",
                    predicted_class
                )

            with result_col2:
                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            st.progress(
                min(max(confidence, 0.0), 1.0)
            )


            # -----------------------------
            # RAG Retrieval
            # -----------------------------
            with st.spinner("Searching knowledge base..."):

                if symptoms.strip():

                    rag_query = (
                        f"Fish classification: {predicted_class}. "
                        f"Observed symptoms: {symptoms}"
                    )

                else:

                    rag_query = (
                        f"Fish classification: {predicted_class}"
                    )

                rag_results = retrieve_information(
                    rag_query,
                    top_k=3
                )


            # -----------------------------
            # Retrieved Knowledge
            # -----------------------------
            st.markdown("### 📚 Retrieved Knowledge")

            if rag_results:

                for i, result in enumerate(
                    rag_results,
                    start=1
                ):

                    with st.expander(
                        f"Knowledge Result {i}"
                    ):
                        st.write(result)

            else:

                st.info(
                    "No relevant information was found "
                    "in the knowledge base."
                )


            # -----------------------------
            # LLM Diagnosis
            # -----------------------------
            disease_prediction = (
                f"Class: {predicted_class}\n"
                f"Confidence: {confidence * 100:.2f}%"
            )

            with st.spinner(
                "Generating diagnosis report..."
            ):

                try:

                    diagnosis = generate_diagnosis(
                        disease_prediction,
                        rag_results
                    )

                    st.markdown("### 🤖 AI-Assisted Diagnosis")

                    st.markdown(
                        '<div class="result-box">',
                        unsafe_allow_html=True
                    )

                    st.markdown(diagnosis)

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )

                except Exception as e:

                    st.error(
                        f"LLM generation failed: {e}"
                    )

else:

    st.info(
        "👆 Please upload a fish image to start diagnosis."
    )
