import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import os
import requests
from bs4 import BeautifulSoup
import wikipedia
import re
import numpy as np
import hashlib
import sentencepiece as spm
import math
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Ensure numpy is available and working
try:
    import numpy as np
    _ = np.array([1,2,3]).tolist()  # Test numpy functionality
except Exception as e:
    raise RuntimeError("Numpy is required and must be working for this script. Please install or fix numpy.") from e

# ======================= DATA COLLECTION =======================

class DataCollector:
    def __init__(self, output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def scrape_wikipedia(self, topics, max_articles_per_topic=50):
        """Scrape Wikipedia articles for given topics"""
        all_texts = []
        
        for topic in tqdm(topics, desc="Scraping Wikipedia"):
            try:
                # Search for articles related to the topic
                search_results = wikipedia.search(topic, results=max_articles_per_topic)
                
                for idx, title in enumerate(search_results[:max_articles_per_topic]):
                    try:
                        page = wikipedia.page(title)
                        content = page.content
                        # Clean and preprocess
                        content = self.clean_text(content)
                        if len(content) > 500:  # Only keep substantial articles
                            all_texts.append({
                                'title': title,
                                'topic': topic,
                                'content': content,
                                'source': 'wikipedia'
                            })
                        # Print progress every 10 articles
                        if (idx + 1) % 10 == 0:
                            print(f"Scraped {idx + 1} articles for topic '{topic}'")
                    except Exception as e:
                        continue
                        
            except Exception as e:
                print(f"Error scraping topic {topic}: {e}")
                continue
                
        return all_texts
    
    def scrape_web_content(self, urls):
        """Scrape content from web URLs"""
        all_texts = []
        
        for url in tqdm(urls, desc="Scraping web content"):
            try:
                response = requests.get(url, timeout=10)
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text
                text = soup.get_text()
                text = self.clean_text(text)
                
                if len(text) > 500:
                    all_texts.append({
                        'url': url,
                        'content': text,
                        'source': 'web'
                    })
                    
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
                
        return all_texts
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\(\)"\']', '', text)
        # Remove very short lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        return '\n'.join(lines)
    
    def save_raw_data(self, data, filename="raw_data.json"):
        """Save collected data"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} documents to {filepath}")

# ======================= DATA PREPROCESSING =======================

class DataPreprocessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.hashes = set()
        
    def load_raw_data(self, filename="raw_data.json"):
        """Load raw collected data"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def deduplicate_exact(self, texts):
        """Remove exact duplicates using hashing"""
        unique_texts = []
        seen_hashes = set()
        
        for text_data in tqdm(texts, desc="Deduplicating"):
            content = text_data['content']
            text_hash = hashlib.md5(content.encode()).hexdigest()
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text_data)
                
        print(f"Removed {len(texts) - len(unique_texts)} exact duplicates")
        return unique_texts
    
    def deduplicate_near(self, texts, similarity_threshold=0.8):
        """Remove near duplicates using MinHash-like approach"""
        def get_shingles(text, k=3):
            """Generate k-shingles from text"""
            words = text.lower().split()
            shingles = set()
            for i in range(len(words) - k + 1):
                shingle = ' '.join(words[i:i+k])
                shingles.add(shingle)
            return shingles
        
        def jaccard_similarity(set1, set2):
            """Calculate Jaccard similarity"""
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0
        
        unique_texts = []
        
        for i, text_data in enumerate(tqdm(texts, desc="Near-duplicate removal")):
            content = text_data['content']
            shingles = get_shingles(content)
            
            is_duplicate = False
            for existing_data in unique_texts:
                existing_shingles = get_shingles(existing_data['content'])
                similarity = jaccard_similarity(shingles, existing_shingles)
                
                if similarity > similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text_data)
                
        print(f"Removed {len(texts) - len(unique_texts)} near duplicates")
        return unique_texts
    
    def quality_filter(self, texts):
        """Filter low-quality texts"""
        filtered_texts = []
        
        for text_data in tqdm(texts, desc="Quality filtering"):
            content = text_data['content']
            
            # Basic quality checks
            if len(content) < 100:  # Too short
                continue
            if len(content) > 50000:  # Too long
                continue
            
            # Check for reasonable word count
            words = content.split()
            if len(words) < 20:
                continue
            
            # Check for reasonable sentence structure
            sentences = content.split('.')
            avg_sentence_length = len(words) / len(sentences) if len(sentences) > 0 else 0
            if avg_sentence_length < 3 or avg_sentence_length > 100:
                continue
            
            # Check character diversity
            unique_chars = len(set(content.lower()))
            if unique_chars < 10:  # Too repetitive
                continue
            
            filtered_texts.append(text_data)
            
        print(f"Kept {len(filtered_texts)} high-quality texts")
        return filtered_texts
    
    def save_processed_data(self, data, filename="processed_data.json"):
        """Save processed data"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} processed documents to {filepath}")

# ======================= TOKENIZATION =======================

class TokenizerTrainer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size
        self.tokenizer = None
    
    def train_tokenizer(self, texts, model_prefix="tinyllama_tokenizer"):
        """Train SentencePiece tokenizer"""
        # Combine all texts into a single file
        combined_text = ""
        for text_data in texts:
            combined_text += text_data['content'] + "\n"
        
        # Save to temporary file
        temp_file = "temp_training_data.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(combined_text)
        
        # Train SentencePiece
        spm.SentencePieceTrainer.train(
            input=temp_file,
            model_prefix=model_prefix,
            vocab_size=self.vocab_size,
            model_type='bpe',
            normalization_rule_name='nfkc',
            remove_extra_whitespaces=True,
            input_sentence_size=2000000,
            shuffle_input_sentence=True,
            pad_id=0,
            eos_id=1,
            unk_id=2,
            bos_id=3,
        )
        
        # Load trained tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(f"{model_prefix}.model")
        
        # Clean up
        os.remove(temp_file)
        
        print(f"Trained tokenizer with vocab size: {self.tokenizer.vocab_size()}")
        return self.tokenizer

# ======================= MODEL ARCHITECTURE =======================

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        sinusoid_inp = torch.einsum("i,j->ij", t, self.inv_freq)
        
        sin = sinusoid_inp.sin()[None, None, :, :]
        cos = sinusoid_inp.cos()[None, None, :, :]
        
        return sin, cos

def apply_rotary_pos_emb(q, k, sin, cos):
    # q, k: [B, n_heads, T, head_dim]
    # sin, cos: [T, head_dim // 2]
    # Apply rotary only to first half of head_dim
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1,1,T,head_dim//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    q1_rot = q1 * cos + rotate_half(q1) * sin
    k1_rot = k1 * cos + rotate_half(k1) * sin
    q_rot = torch.cat([q1_rot, q2], dim=-1)
    k_rot = torch.cat([k1_rot, k2], dim=-1)
    return q_rot, k_rot

def rotate_half(x):
    # x: [..., head_dim//2]
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        sin, cos = self.rotary_emb(x, T)  # sin, cos: [1,1,T,head_dim//2]
        sin = sin.squeeze(0).squeeze(0)  # [T, head_dim//2]
        cos = cos.squeeze(0).squeeze(0)
        q, k = apply_rotary_pos_emb(q, k, sin, cos)
        
        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if mask is not None:
            att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        
        return y

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.feed_forward = SwiGLU(dim, 4 * dim)
        self.attention_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        x = x + self.dropout(self.attention(self.attention_norm(x), mask))
        # Feed-forward with residual connection
        x = x + self.dropout(self.feed_forward(self.ffn_norm(x)))
        return x

class TinyLLaMA(nn.Module):
    def __init__(self, vocab_size, dim=512, n_layers=8, n_heads=8, max_seq_len=1024, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Token embeddings
        x = self.token_embedding(idx)
        
        # Create causal mask
        mask = torch.tril(torch.ones(T, T, device=idx.device)).view(1, 1, T, T)
        
        # Apply transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Crop idx to max_seq_len
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]
                
                # Forward pass
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :] / temperature
                
                # Top-k filtering
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# ======================= DATASET =======================

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]['content']
        
        # Tokenize
        tokens = self.tokenizer.encode(text, out_type=int)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_id()] * (self.max_length - len(tokens))
        
        # Convert to tensor
        x = torch.tensor(tokens[:-1], dtype=torch.long)
        y = torch.tensor(tokens[1:], dtype=torch.long)
        
        return x, y

# ======================= TRAINING =======================

class Trainer:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def train(self, train_data, val_data, epochs=10, batch_size=8, lr=3e-4, save_every=1000):
        # Create datasets
        train_dataset = TextDataset(train_data, self.tokenizer)
        val_dataset = TextDataset(val_data, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Optimizer with AdamW
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.1)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * epochs)
        
        # Training loop
        self.model.train()
        global_step = 0
        
        for epoch in range(epochs):
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                logits, loss = self.model(x, y)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                global_step += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
                
                # Save checkpoint
                if global_step % save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{global_step}.pt")
            
            # Validation
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, loss = self.model(x, y)
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'dim': self.model.dim,
                'max_seq_len': self.model.max_seq_len,
                'n_layers': len(self.model.layers),
                'n_heads': self.model.layers[0].attention.n_heads if self.model.layers else None,
                'dropout': self.model.layers[0].attention.dropout.p if self.model.layers else 0.1,
            }
        }
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {filename}")
        return checkpoint.get('model_config', {})

# ======================= INFERENCE =======================

class TinyLLaMAInference:
    def __init__(self, model_path, tokenizer_path, device='cuda'):
        self.device = device
        
        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.load(tokenizer_path)
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['model_config']
        
        self.model = TinyLLaMA(
            vocab_size=config['vocab_size'],
            dim=config['dim'],
            n_layers=config.get('n_layers', 8),
            n_heads=config.get('n_heads', 8),
            max_seq_len=config['max_seq_len'],
            dropout=config.get('dropout', 0.1)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
    
    def generate_text(self, prompt, max_length=100, temperature=0.7, top_k=50):
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt, out_type=int)
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        
        # Generate
        generated = self.model.generate(tokens, max_length, temperature, top_k)
        
        # Decode
        # Use .tolist() directly to avoid numpy dependency
        generated_tokens = generated[0].cpu().tolist()
        response = self.tokenizer.decode(generated_tokens)
        
        return response

# ======================= MAIN EXECUTION =======================

def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Topics to search (AI only, very large dataset)
    topics = [
        "artificial intelligence"
    ]
    
    print("=== PHASE 1: DATA COLLECTION ===")
    collector = DataCollector()
    
    # Collect Wikipedia data (very large for AI)
    wiki_data = collector.scrape_wikipedia(topics, max_articles_per_topic=2000)
    print(f"Number of Wikipedia sources collected: {len(wiki_data)}")
    collector.save_raw_data(wiki_data, "wiki_data.json")
    
    print("=== PHASE 2: DATA PREPROCESSING ===")
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    raw_data = preprocessor.load_raw_data("wiki_data.json")
    
    # Deduplicate and filter
    deduped_data = preprocessor.deduplicate_exact(raw_data)
    near_deduped_data = preprocessor.deduplicate_near(deduped_data, similarity_threshold=0.7)
    clean_data = preprocessor.quality_filter(near_deduped_data)
    
    preprocessor.save_processed_data(clean_data, "clean_data.json")
    
    print("=== PHASE 3: TOKENIZER TRAINING ===")
    tokenizer_trainer = TokenizerTrainer(vocab_size=12000)  # Smaller vocab for TinyLLaMA
    tokenizer = tokenizer_trainer.train_tokenizer(clean_data, "tinyllama_tokenizer")
    
    print("=== PHASE 4: MODEL TRAINING ===")
    # Split data
    train_size = int(0.9 * len(clean_data))
    train_data = clean_data[:train_size]
    val_data = clean_data[train_size:]
    
    # Create model
    model = TinyLLaMA(
        vocab_size=tokenizer.vocab_size(),
        dim=512,        # Increase dimension
        n_layers=8,     # More layers
        n_heads=8,      # More heads
        max_seq_len=512,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = Trainer(model, tokenizer, device)
    trainer.train(train_data, val_data, epochs=100, batch_size=8, lr=5e-4)  # Train for 100 epochs
    
    # Save final model
    trainer.save_checkpoint("tinyllama_final.pt")
    
    print("=== PHASE 5: INFERENCE TEST ===")
    # Test inference
    inference = TinyLLaMAInference("tinyllama_final.pt", "tinyllama_tokenizer.model", device)
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Machine learning algorithms are",
        "In computer science, we study"
    ]
    
    for prompt in test_prompts:
        response = inference.generate_text(prompt, max_length=50, temperature=1.0, top_k=100)
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("-" * 50)
    
    print("=== TRAINING COMPLETE ===")
    print("Model saved as: tinyllama_final.pt")
    print("Tokenizer saved as: tinyllama_tokenizer.model")

if __name__ == "__main__":
    main()