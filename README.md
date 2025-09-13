# GPT-Style Transformer Language Model

A PyTorch implementation of a GPT-style decoder-only transformer for language modeling, trained on Wikipedia data with ByteLevelBPE tokenization.

## Features

- **Modern Architecture**: Decoder-only transformer with multi-head attention and feed-forward networks
- **Efficient Tokenization**: ByteLevelBPE tokenizer with 30,000 vocabulary size
- **Advanced Sampling**: Support for greedy, top-k, and nucleus (top-p) sampling
- **KV Caching**: Efficient autoregressive generation with key-value caching
- **Regularization**: Comprehensive dropout, layer normalization, and weight decay
- **Optimized Training**: AdamW optimizer with learning rate scheduling and gradient clipping

## Architecture Details

### Model Components

- **Multi-Head Attention**: Custom implementation with causal masking and attention dropout
- **Position Encoding**: Sinusoidal positional embeddings
- **Feed-Forward Network**: Two-layer MLP with ReLU activation and dropout
- **Layer Normalization**: Pre-norm architecture for stable training
- **Residual Connections**: Skip connections around attention and FFN layers

### Default Configuration

```python
embed_size = 128        # Embedding dimension
num_heads = 4           # Number of attention heads  
depth = 4               # Number of transformer layers
vocab_size = 30000      # ByteLevelBPE vocabulary size
max_seq_len = 2050      # Maximum sequence length
dropout = 0.6           # Dropout probability
```

## Requirements

```
torch>=1.9.0
tokenizers>=0.13.0
nltk>=3.7
```

Install dependencies:
```bash
pip install torch tokenizers nltk
```

## Dataset

The model is trained on Wikipedia text data. The expected data structure:

```
data/
├── wiki.train.tokens/
│   └── wiki.train.tokens
└── wiki.valid.tokens/
    └── wiki.valid.tokens
```

### Data Format
- Plain text files with one sentence per line
- Articles separated by lines starting with '='
- UTF-8 encoding

## Usage

### Training

```python
# Initialize model
model = Transformer(
    voc_size=30000,
    embed_size=128,
    num_heads=4,
    depth=4,
    pad_idx=2,
    p=0.6
).to(device)

# Setup training
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)

# Train
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, criterion)
    val_loss = evaluate(model, val_loader, criterion)
```

### Text Generation

```python
# Load trained model
model = Transformer(voc_size=30000, embed_size=128, num_heads=4, depth=4, pad_idx=2)
model.load_state_dict(torch.load("models/best_model.pth"))

# Generate text
prompt = "Hello world"
prompt_ids = tokenizer.encode(prompt).ids
generated_text = model.generate_from_prompt(
    prompt_ids, 
    sample='p',      # 'greedy', 'k', or 'p'
    max_size=100
)
print(generated_text)
```

### Sampling Methods

1. **Greedy Sampling**: Always selects the most probable next token
2. **Top-K Sampling**: Samples from the top-k most probable tokens (k=50)
3. **Nucleus (Top-P) Sampling**: Samples from tokens whose cumulative probability ≥ p (p=0.9)

## Training Configuration

### Hyperparameters

- **Learning Rate**: 1e-4 with linear warmup and decay
- **Batch Size**: 16
- **Weight Decay**: 0.1
- **Gradient Clipping**: 0.5
- **Label Smoothing**: 0.05
- **Early Stopping**: Patience of 5 epochs

### Regularization Techniques

- Multiple dropout layers (embedding, attention, FFN, final)
- Layer normalization
- Weight decay
- Label smoothing
- Gradient clipping

## File Structure

```
├── byte_token.ipynb          # Main training notebook
├── models/                   # Saved model checkpoints
│   ├── best_model.pth
│   └── overall_best.pth
├── data/                     # Training data
│   ├── wiki.train.tokens/
│   └── wiki.valid.tokens/
└── README.md
```

## Model Architecture

```
Transformer(
  (embed): Embedding(30000, 128)
  (embed_dropout): Dropout(p=0.3)
  (pe): PositionalEncoding(
    (dropout): Dropout(p=0.6)
  )
  (dropout): Dropout(p=0.6)
  (heads): ModuleList(
    (0-3): 4 x ModuleList(
      (0): MultiAttentionHead(
        (norm): LayerNorm((128,))
        (attention_dropout): Dropout(p=0.1)
      )
      (1): LayerNorm((128,))
      (2): FFN(
        (layers): Sequential(...)
      )
      (3): LayerNorm((128,))
    )
  )
  (final_dropout): Dropout(p=0.3)
  (linear): Linear(in_features=128, out_features=30000, bias=False)
)
```

## Key Features

### Efficient Generation
- KV-cache implementation for fast autoregressive generation
- Support for multiple sampling strategies
- Automatic early stopping on EOS token

### Training Optimizations
 - Comprehensive logging and checkpointing
- Overfitting prevention with multiple regularization techniques

### Data Processing
- ByteLevelBPE tokenization with special tokens (`<SOS>`, `<EOS>`, `<PAD>`, `<UNK>`)
- Dynamic padding within batches
- Causal masking for autoregressive training

## Performance Considerations
- Memory usage: Optimized with KV caching
- Training time: Varies by dataset size and hardware
- Generation speed: Fast with cached attention

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or sequence length
2. **Slow Training**: Check data loading efficiency and use GPU
3. **Poor Generation**: Adjust sampling parameters or train longer
4. **Overfitting**: Increase dropout, reduce model size, or add more data

### Debugging Tips

- Monitor train/validation loss curves
- Check attention patterns during generation
- Validate tokenizer behavior with sample texts
- Use gradient clipping if gradients explode

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Transformer architecture based on "Attention Is All You Need"
- GPT-style implementation following OpenAI's approach
- ByteLevelBPE tokenization inspired by GPT-2

---

**Note**: This implementation is designed for educational and research purposes. For production use, consider using established frameworks like Hugging Face Transformers.
