import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = (
    "/content/drive/MyDrive/Defensive-AI-Cybersecurity/"
    "transformer_artifacts/distilbert_phishing"
)

st.set_page_config(
    page_title="Defensive AI Phishing Detector",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Defensive AI Phishing Detector")

st.write(
    "A research prototype that uses a trained DistilBERT model "
    "to classify email text as legitimate or potential phishing."
)

st.info(
    "This is an experimental defensive cybersecurity tool. "
    "Predictions may be incorrect and should not be treated as "
    "a replacement for professional security systems."
)


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


tokenizer, model, device = load_model()

email_text = st.text_area(
    "Paste email text below:",
    height=250,
    placeholder="Paste an email here for analysis..."
)

if st.button("Analyze Email", type="primary"):

    if not email_text.strip():
        st.warning("Please enter some email text before analyzing.")

    else:
        inputs = tokenizer(
            email_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        legitimate_probability = float(
            probabilities[0].item()
        )

        phishing_probability = float(
            probabilities[1].item()
        )

        prediction = int(
            torch.argmax(probabilities).item()
        )

        st.divider()

        if prediction == 1:
            st.error("Potential Phishing Detected")
        else:
            st.success("Likely Legitimate")

        st.metric(
            "Phishing Probability",
            f"{phishing_probability:.1%}"
        )

        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.1%}"
        )

        st.caption(
            "Model: DistilBERT | Maximum input length: 256 tokens"
        )

        st.warning(
            "Model confidence is not a guarantee of correctness. "
            "The research evaluation identified high-confidence "
            "errors, particularly on some synthetic email styles."
        )

st.divider()

st.caption(
    "Defensive AI Cybersecurity Research Project | "
    "Human-Layer Phishing Detection Prototype"
)
