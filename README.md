# TinyLLaMA: Custom Language Model Training Pipeline

A complete end-to-end pipeline for training a small-scale language model from scratch, including data collection, preprocessing, tokenization, and training with a LLaMA-inspired architecture.

## 🌟 Features

- **Automated Data Collection**: Scrapes Wikipedia articles and web content
- **Robust Data Preprocessing**: Deduplication, quality filtering, and text cleaning
- **Custom Tokenizer Training**: Uses SentencePiece with BPE encoding
- **Modern Architecture**: Implements LLaMA-inspired features including:
  - RMSNorm for layer normalization
  - Rotary Position Embeddings (RoPE)
  - SwiGLU activation function
  - Multi-head attention with causal masking
- **Training Pipeline**: Complete training loop with validation and checkpointing
- **Inference Engine**: Ready-to-use text generation interface

## 📋 Requirements

### Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy
pip install sentencepiece
pip install wikipedia-api
pip install beautifulsoup4
pip install requests
pip install tqdm
```

### System Requirements

- **GPU**: CUDA-compatible GPU recommended (training will be very slow on CPU)
- **RAM**: At least 8GB RAM, 16GB+ recommended
- **Storage**: 5-10GB free space for data and model checkpoints
- **Python**: 3.8 or higher

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

```bash
python training.py
```

This will execute all phases:
1. Data collection from Wikipedia
2. Data preprocessing and cleaning
3. Tokenizer training
4. Model training
5. Inference testing

### 3. Monitor Training Progress

The script will show progress bars and training metrics:
- Loss values during training
- Learning rate schedules
- Validation performance
- Checkpoint saving

## 📊 Pipeline Overview

### Phase 1: Data Collection
- Scrapes Wikipedia articles related to AI topics
- Collects up to 2000 articles per topic
- Supports custom topic lists and web URL scraping
- Saves raw data in JSON format

### Phase 2: Data Preprocessing
- **Exact Deduplication**: Removes identical content using MD5 hashing
- **Near Deduplication**: Uses Jaccard similarity on text shingles
- **Quality Filtering**: Removes low-quality content based on:
  - Text length (100-50,000 characters)
  - Word count and sentence structure
  - Character diversity

### Phase 3: Tokenizer Training
- Trains SentencePiece tokenizer with BPE algorithm
- Vocabulary size: 12,000 tokens
- Includes special tokens: PAD, EOS, UNK, BOS

### Phase 4: Model Architecture

```
TinyLLaMA Model Specifications:
├── Embedding Dimension: 512
├── Number of Layers: 8
├── Attention Heads: 8
├── Vocabulary Size: 12,000
├── Max Sequence Length: 512
├── Parameters: ~15M
└── Features:
    ├── RMSNorm
    ├── Rotary Position Embeddings
    ├── SwiGLU Activation
    └── Causal Self-Attention
```

### Phase 5: Training Configuration
- **Optimizer**: AdamW with weight decay (0.1)
- **Learning Rate**: 5e-4 with cosine annealing
- **Batch Size**: 8
- **Epochs**: 100
- **Gradient Clipping**: Max norm 1.0
- **Validation Split**: 90/10 train/validation

## 🔧 Configuration

### Customizing Topics

Edit the `topics` list in `main()`:

```python
topics = [
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural networks"
]
```

### Model Hyperparameters

Modify the model configuration:

```python
model = TinyLLaMA(
    vocab_size=tokenizer.vocab_size(),
    dim=512,        # Embedding dimension
    n_layers=8,     # Number of transformer layers
    n_heads=8,      # Number of attention heads
    max_seq_len=512, # Maximum sequence length
    dropout=0.1     # Dropout rate
)
```

### Training Parameters

Adjust training settings:

```python
trainer.train(
    train_data, 
    val_data, 
    epochs=100,      # Number of training epochs
    batch_size=8,    # Batch size
    lr=5e-4         # Learning rate
)
```

## 📁 Output Files

After training, you'll have:

```
├── data/
│   ├── wiki_data.json          # Raw scraped data
│   └── clean_data.json         # Preprocessed data
├── tinyllama_tokenizer.model   # Trained tokenizer
├── tinyllama_tokenizer.vocab   # Tokenizer vocabulary
├── tinyllama_final.pt          # Final trained model
└── checkpoint_*.pt             # Training checkpoints
```

## 🎯 Usage Examples

### Text Generation

```python
from training import TinyLLaMAInference

# Load trained model
inference = TinyLLaMAInference(
    model_path="tinyllama_final.pt",
    tokenizer_path="tinyllama_tokenizer.model"
)

# Generate text
response = inference.generate_text(
    prompt="The future of artificial intelligence is",
    max_length=100,
    temperature=0.7,
    top_k=50
)

print(response)
```

### Custom Data Training

```python
# Use your own data
custom_data = [
    {"content": "Your text data here...", "source": "custom"},
    {"content": "More text data...", "source": "custom"}
]

# Train with custom data
trainer = Trainer(model, tokenizer, device)
trainer.train(custom_data, val_data, epochs=50)
```

## 🛠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**
- Reduce batch size: `batch_size=4` or `batch_size=2`
- Reduce sequence length: `max_seq_len=256`
- Use gradient accumulation

**Slow Training**
- Ensure you're using GPU: Check `torch.cuda.is_available()`
- Reduce model size: `dim=256, n_layers=4`
- Use mixed precision training

**Poor Generation Quality**
- Train for more epochs
- Increase model size
- Improve data quality
- Adjust generation parameters (temperature, top_k)

### Memory Optimization

For limited GPU memory:

```python
# Smaller model configuration
model = TinyLLaMA(
    vocab_size=tokenizer.vocab_size(),
    dim=256,        # Reduced dimension
    n_layers=4,     # Fewer layers
    n_heads=4,      # Fewer heads
    max_seq_len=256, # Shorter sequences
    dropout=0.1
)
```

## 📈 Performance Expectations

### Training Time
- **GPU (RTX 3080)**: ~2-4 hours for 100 epochs
- **GPU (GTX 1060)**: ~8-12 hours for 100 epochs  
- **CPU**: Not recommended (days/weeks)

### Model Quality
- **Small Dataset**: Basic coherence, limited knowledge
- **Large Dataset**: Better coherence, domain knowledge
- **Extended Training**: Improved fluency and consistency

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by the LLaMA architecture from Meta AI
- Uses SentencePiece tokenization
- Built with PyTorch framework
- Wikipedia for providing open data

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the code comments for implementation details

---

**Note**: This is an educational project for learning language model training. For production use, consider using established frameworks like Transformers or fine-tuning existing models.
