# Transformer Training in Google Colab

Use this setup for the advanced NLP comparison.

## Recommended Colab setup

1. Open a new Google Colab notebook.
2. Set **Runtime > Change runtime type > GPU**.
3. Clone this repository.
4. Install the transformer requirements.
5. Make sure the processed training and validation CSV files are available in:
   - `Phase-1-Phishing-Detection/data/processed/train.csv`
   - `Phase-1-Phishing-Detection/data/processed/validation.csv`
6. Run `src/train_transformer.py`.

The script intentionally does **not** read `test.csv`.

## Training configuration

- Model: `distilbert-base-uncased`
- Maximum sequence length: 256 tokens
- Epochs: 2
- Train batch size: 16
- Validation batch size: 32
- Learning rate: 2e-5
- Best checkpoint selected by validation F1

## Important

The processed CSV files are intentionally not stored in GitHub if they are ignored by `.gitignore`. They must be uploaded or otherwise made available inside the Colab runtime before training.

Actual transformer training will take longer than one minute and may take tens of minutes depending on the GPU assigned by Colab.
