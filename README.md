# Transformer Language Model

Minimal Transformer-based language model built from scratch using PyTorch.

## Objectives
- Implement a simple Transformer encoder-decoder stack for language modeling
- Support autoregressive generation with cached keys/values
- Provide top-k and greedy sampling methods

## Features
- **Embedding + Positional Encoding**
- **Multi-Head Attention** (`MultiAttentionHead`) with causal masking and cache
- **Feed-Forward Layers** with residual connections & LayerNorm
- **Training Loop** using PyTorch optimizers and loss functions
- **Text Generation** from a given prompt

## Usage
```python
model.eval()
generated = model.generate_from_prompt("Once upon a time", sample="k")
print(generated)
