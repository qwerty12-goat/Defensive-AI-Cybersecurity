# Phase 1 Dataset Documentation

This folder documents the datasets used for the Phase 1 phishing detection research project.

## Dataset Requirements

The initial dataset should include:

- Legitimate emails
- Traditional phishing emails
- Email text content, not only URLs
- Reliable labels identifying legitimate and phishing messages
- A public source suitable for research use

AI-generated phishing examples will be added in a later stage after the baseline dataset is established.

## Dataset Sources

### Dataset 1

- **Name:** To be selected
- **Source:** To be selected
- **Date accessed:** To be added
- **Original size:** To be added
- **Labels/categories:** To be added
- **License or usage restrictions:** To be added
- **How it will be used:** Baseline training and evaluation for legitimate vs. traditional phishing email detection

## Folder Structure

- `raw/` stores original downloaded data
- `processed/` stores cleaned and transformed datasets used for modeling

## Research Notes

The final test set should remain isolated from training and validation data. Duplicate and near-duplicate emails should be identified to reduce data leakage.
