# Transformer & BERT Models from Scratch (PyTorch)

This project showcases two key components built entirely from scratch using PyTorch:

1. **BERT Model** — A transformer-based encoder architecture tailored for natural language understanding tasks.
2. **English-to-Italian Language Transformer** — A complete encoder-decoder transformer model for machine translation.


## Directory

This directory contains two major implementations:

- `BERT/` – A self-contained BERT model with tokenization, embedding, and encoder blocks.
- `Transformer/` – A transformer model capable of translating English sentences to Italian.

## Key Features

### BERT (Bidirectional Encoder Representations from Transformers)
- Built from scratch with PyTorch
- Token embeddings + positional encodings
- Multi-head self-attention & feed-forward layers
- Masked Language Modeling (MLM) training setup

###  Language Transformer (English → Italian)
- Complete encoder-decoder transformer architecture
- Positional encoding and attention masking
- Trained on a custom English–Italian sentence pair dataset
- Evaluation with BLEU score & inference for test sentences

## Technologies Used
- Python 3.10+
- PyTorch
- NumPy
- Matplotlib (for visualization)
- tqdm (for progress tracking)

## How to Run

Clone the repository:
```bash
git clone https://github.com/Vishnu0107/AI_Project.git
git fetch origin
git checkout vishnu/transformer
git pull origin vishnu/transformer
```
Select the appropriate project you wish to run and execute them
