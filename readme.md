# Seq2Seq Transformer: German-to-English Neural Machine Translation

![Python](https://img.shields.io/badge/Python-3.7+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete implementation of a sequence-to-sequence transformer model for German-to-English neural machine translation (NMT), built from scratch using PyTorch. This project demonstrates core transformer architecture concepts including multi-head attention, positional encoding, and the encoder-decoder paradigm.

## Problem Statement

Language translation remains a critical challenge in natural language processing. This project implements a transformer-based neural machine translation system that can translate German text to English, showcasing how transformers overcome limitations of traditional RNN/LSTM models through parallel processing and improved context understanding.

## Features

- **Transformer Architecture**: Full encoder-decoder implementation with multi-head self-attention
- **Positional Encoding**: Fixed sinusoidal positional embeddings for sequence order information
- **Training Pipeline**: Complete training loop with teacher forcing and loss calculation
- **Inference Methods**: Greedy decoding for efficient translation generation
- **Evaluation Metrics**: BLEU score computation for translation quality assessment
- **PDF Translation**: Batch document translation with error handling
- **Visualization**: Training/validation loss curves and sample translations

## Quick Start

### Prerequisites

```bash
pip install torch==1.9.0
pip install torchtext==0.14.1
pip install torchdata==0.5.1
pip install spacy==3.7.2
pip install nltk==3.8.1
pip install pdfplumber==0.9.0
pip install fpdf==1.7.2
```

### Setup

```bash
# Download language models
python -m spacy download de
python -m spacy download en

# Clone repository and install
git clone <repository-url>
cd translation-transformer
```

### Run

Open `seq2seq_transformer_nmt.ipynb` in Jupyter and run all cells:

```bash
jupyter notebook seq2seq_transformer_nmt.ipynb
```

## Model Architecture

| Component                        | Details                                |
| -------------------------------- | -------------------------------------- |
| **Embedding Dimension**    | 512                                    |
| **Attention Heads**        | 8                                      |
| **Feed-Forward Dimension** | 512                                    |
| **Encoder Layers**         | 3                                      |
| **Decoder Layers**         | 3                                      |
| **Dropout**                | 0.1                                    |
| **Optimizer**              | Adam (lr=0.0001, β₁=0.9, β₂=0.98)  |
| **Loss Function**          | Cross-Entropy (ignores padding tokens) |

### Architecture Overview

```
German Text Input
    ↓
Tokenization & Embedding
    ↓
Positional Encoding
    ↓
┌─────────────────┐
│   Transformer   │
│   Encoder (3x)  │  → Multi-head Attention + Feed-Forward
│                 │
│   Decoder (3x)  │  → Masked Self-Attention + Cross-Attention + Feed-Forward
└─────────────────┘
    ↓
Linear Output Layer (Vocab Size)
    ↓
Softmax & Argmax
    ↓
English Text Output
```

## Project Structure

```
.
├── seq2seq_transformer_nmt.ipynb      # Main notebook
├── Multi30K_de_en_dataloader.py       # Data loading utilities
├── transformer.pt                      # Pre-trained model weights
├── input_de.pdf                       # Sample PDF for translation
├── README.md                          # This file
└── requirements.txt                   # Package dependencies
```

## Key Concepts Implemented

### 1. **Masking**

- **Padding Mask**: Prevents attention to padding tokens
- **Causal Mask**: Prevents decoder from attending to future tokens during training

### 2. **Positional Encoding**

- Fixed sinusoidal embeddings encode token positions
- Maintains sequence order information for attention mechanism

### 3. **Teacher Forcing**

- During training, decoder receives ground-truth target tokens as input
- Enables faster convergence compared to autoregressive training

### 4. **Greedy Decoding**

- During inference, decoder uses its own predictions as next input
- Generates translation token-by-token until EOS token

### 5. **Multi-Head Attention**

- 8 parallel attention heads capture different aspects of input
- Each head attends to different parts of the input sequence

## Dataset

The model trains on the **Multi30K dataset** (English-German parallel corpus):

- **Training**: ~30,000 sentence pairs
- **Validation**: ~1,000 sentence pairs
- **Vocabulary**: ~10,000 tokens per language
- **Download**: Automatic via torchtext in the notebook

### Special Tokens

| Token     | Index | Purpose                               |
| --------- | ----- | ------------------------------------- |
| `<unk>` | 0     | Unknown words                         |
| `<pad>` | 1     | Padding for batch alignment           |
| `<bos>` | 2     | Beginning of sequence (decoder start) |
| `<eos>` | 3     | End of sequence (decoder stop)        |

## Training

### Configuration

- **Batch Size**: 128
- **Epochs**: 10-40 (depending on computational resources)
- **Learning Rate**: 0.0001 with Adam optimizer
- **Loss**: Cross-entropy (padding tokens ignored)

### Training Time

- **GPU**: ~2-3 hours for 10 epochs
- **CPU**: ~40-60 minutes per epoch (use pre-trained weights provided)

### Using Pre-trained Weights

The notebook includes a pre-trained model (`transformer.pt`) trained for 40 epochs. To load it:

```python
transformer.load_state_dict(
    torch.load('transformer.pt', map_location=torch.device('cpu'))
)
```

## Inference & Evaluation

### Translation Example

```python
from translate import translate

german_text = "Ein brauner Hund spielt im Schnee."
english_translation = translate(transformer, german_text)
print(english_translation)
# Output: "A brown dog is playing in the snow ."
```

### BLEU Score

Evaluate translation quality using BLEU (Bilingual Evaluation Understudy):

```python
from nltk.translate.bleu_score import sentence_bleu

reference = "A brown dog is playing in the snow ."
hypothesis = "A brown dog plays in snow ."

bleu = sentence_bleu([reference.split()], hypothesis.split())
print(f"BLEU Score: {bleu:.4f}")
```

### PDF Translation

Translate entire documents:

```python
translate_pdf(
    input_file='document_de.pdf',
    translator_model=transformer,
    output_file='document_en.pdf'
)
```

## Results

The trained model achieves:

- **Validation Loss**: ~4.5-5.0 after 10 epochs
- **Sample Quality**: Good grammatical structure with minor vocabulary differences
- **Inference Speed**: ~100-200ms per sentence (CPU)

See notebook for training curves and qualitative translation examples.

## File Descriptions

### Core Components

- **PositionalEncoding**: Adds position information to embeddings
- **TokenEmbedding**: Converts token indices to dense vectors
- **Seq2SeqTransformer**: Main model combining encoder, decoder, and output layer
- **greedy_decode()**: Inference function for autoregressive translation
- **translate()**: User-friendly translation wrapper

### Utilities

- **create_mask()**: Generates attention masks for encoder/decoder
- **generate_square_subsequent_mask()**: Creates causal mask for autoregressive decoding
- **evaluate()**: Computes loss on validation set
- **train_epoch()**: Single training iteration
- **translate_pdf()**: Batch PDF document translation with error handling

## Error Handling & Robustness

The implementation includes:

- ✓ Input validation (file existence, path checks)
- ✓ Error recovery (graceful handling of translation failures)
- ✓ Encoding safety (handles special characters)
- ✓ Progress tracking (tqdm progress bars)
- ✓ Comprehensive logging (detailed error messages)

## Skills Demonstrated

- Deep learning architecture implementation
- Attention mechanisms and transformer models
- Natural language processing (tokenization, vocabulary building)
- PyTorch and neural network training
- Encoder-decoder paradigms
- Sequence-to-sequence models
- Evaluation metrics and model assessment
- Production-ready error handling

## Common Issues & Solutions

### Out of Memory Error

- Reduce batch size from 128 to 64 or 32
- Use a smaller model (fewer layers/heads)
- Run on GPU instead of CPU

### Poor Translation Quality

- Train for more epochs (40 instead of 10)
- Use the pre-trained weights provided
- Ensure correct tokenization and vocabulary

### Slow Training

- Use GPU (CUDA or Metal)
- Reduce validation frequency
- Use the pre-trained model for inference

## License

MIT

## Acknowledgments

- Project completed as part of IBM AI Engineering Professional Certificate
- Multi30K Dataset: Research community
- PyTorch & TorchText: Meta AI
