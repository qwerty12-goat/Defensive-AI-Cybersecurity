# Defensive AI Phishing Detector Demo

This directory contains the interactive Streamlit prototype for the human-layer phishing detection research project.

The application accepts email text and uses the trained DistilBERT classifier to estimate whether the message is legitimate or belongs to the phishing class.

## Model

The demo loads the trained model from Hugging Face:

`ozuvadh/defensive-ai-phishing-distilbert`

Model page: https://huggingface.co/ozuvadh/defensive-ai-phishing-distilbert

The classifier uses a maximum input length of 256 tokens.

## Run the Demo

From this directory, install the required packages:

```bash
pip install -r requirements.txt
```

Then start the application:

```bash
streamlit run phishing_detector_app.py
```

Streamlit will display the local address where the application is running.

The first launch may take longer because the trained DistilBERT model must be downloaded and cached.

## Interface

The application displays:

- a binary classification: likely legitimate or potential phishing;
- phishing probability;
- legitimate probability;
- the model used for inference;
- a warning explaining that model confidence does not guarantee correctness.

## Demonstration Checks

During prototype testing, three controlled examples were used to verify the interface and illustrate model behavior:

| Example type | Demo result |
| --- | --- |
| Routine legitimate message | Likely Legitimate, approximately 100% legitimate probability |
| Security-style phishing-class message | Potential Phishing Detected, 99.9% phishing probability |
| Workplace-style phishing-class message | Likely Legitimate, approximately 100% legitimate probability |

The third result is intentionally documented because it demonstrates a known limitation rather than only showing successful classifications. In the separate 500-message synthetic evaluation, workplace/business-style messages were also difficult for the DistilBERT classifier.

These three demonstration checks are illustrative examples, not an additional benchmark.

## Research Prototype Warning

This application is an experimental defensive cybersecurity research prototype. It should not be used as a replacement for a production email-security system.

The research found that high model confidence does not guarantee correctness. In particular, the DistilBERT model produced high-confidence errors on some controlled synthetic email styles.

Do not submit private, confidential, or sensitive email content to an untrusted deployment of this application.

## Reproducibility

The model is loaded directly from Hugging Face rather than from a researcher-specific local or Google Drive path. This allows the same application code to be run on another compatible machine with internet access.

For the full experimental methodology, model evaluation, robustness analysis, and limitations, see the main project documentation.
