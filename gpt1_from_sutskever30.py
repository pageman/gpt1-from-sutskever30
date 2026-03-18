"""
GPT-1 Implementation Constructed from Sutskever-30 Papers
=========================================================

This implementation constructs GPT-1 ("Improving Language Understanding 
by Generative Pre-Training" by OpenAI, 2018) using code adapted from:

- Paper 02: The Unreasonable Effectiveness of RNNs (text generation, sampling)
- Paper 13: Attention Is All You Need (Transformer architecture core)
- Paper 27: Multi-token Prediction (next-token prediction concepts)

GPT-1 is essentially a Transformer Decoder with:
1. Token embeddings + Positional embeddings
2. Stacked Transformer decoder blocks (masked self-attention + FFN)
3. Language modeling head (predict next token)
4. Pre-training objective: next-token prediction

All implementations use NumPy only for educational clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import re

np.random.seed(42)

# =============================================================================
# COMPONENTS FROM PAPER 13: ATTENTION IS ALL YOU NEED
# =============================================================================

def softmax(x, axis=-1):
    """
    Numerically stable softmax
    [From Paper 13: Attention Is All You Need]
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention
    
    [From Paper 13: Attention Is All You Need]
    
    Q: Queries (seq_len_q, d_k)
    K: Keys (seq_len_k, d_k)  
    V: Values (seq_len_v, d_v)
    mask: Optional mask (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    # Apply mask if provided (for causality or padding)
    if mask is not None:
        scores = scores + (mask * -1e9)
    
    # Softmax to get attention weights
    attention_weights = softmax(scores, axis=-1)
    
    # Weighted sum of values
    output = np.dot(attention_weights, V)
    
    return output, attention_weights


def create_causal_mask(seq_len):
    """
    Create mask to prevent attending to future positions
    
    [From Paper 13: Attention Is All You Need]
    Critical for autoregressive generation in GPT
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask


def positional_encoding(seq_len, d_model):
    """
    Create sinusoidal positional encoding
    
    [From Paper 13: Attention Is All You Need]
    
    Since Transformers have no recurrence, we add position information:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # Apply sin to even indices
    pe[:, 0::2] = np.sin(position * div_term)
    
    # Apply cos to odd indices
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(position * div_term)
    else:
        pe[:, 1::2] = np.cos(position * div_term[:-1])
    
    return pe


class MultiHeadAttention:
    """
    Multi-Head Attention Mechanism
    
    [From Paper 13: Attention Is All You Need]
    
    MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V for all heads (parallelized)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.1
        
        # Cache for backward pass
        self.cache = {}
    
    def split_heads(self, x):
        """Split into multiple heads: (seq_len, d_model) -> (num_heads, seq_len, d_k)"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)
    
    def combine_heads(self, x):
        """Combine heads: (num_heads, seq_len, d_k) -> (seq_len, d_model)"""
        seq_len = x.shape[1]
        x = x.transpose(1, 0, 2)
        return x.reshape(seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """
        Multi-head attention forward pass
        
        Q, K, V: (seq_len, d_model)
        """
        # Store for backward
        self.cache['Q_input'] = Q
        self.cache['K_input'] = K
        self.cache['V_input'] = V
        self.cache['mask'] = mask
        
        # Linear projections
        Q_proj = np.dot(Q, self.W_q.T)
        K_proj = np.dot(K, self.W_k.T)
        V_proj = np.dot(V, self.W_v.T)
        
        # Split into multiple heads
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)
        
        # Apply attention to each head
        head_outputs = []
        self.attention_weights = []
        
        for i in range(self.num_heads):
            head_out, head_attn = scaled_dot_product_attention(
                Q_heads[i], K_heads[i], V_heads[i], mask
            )
            head_outputs.append(head_out)
            self.attention_weights.append(head_attn)
        
        # Stack heads
        heads = np.stack(head_outputs, axis=0)
        
        # Combine heads
        combined = self.combine_heads(heads)
        
        # Final linear projection
        output = np.dot(combined, self.W_o.T)
        
        return output


class LayerNorm:
    """
    Layer Normalization
    
    [From Paper 13: Attention Is All You Need]
    Normalize across features (not batch like BatchNorm)
    """
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        
        normalized = (x - mean) / (std + self.eps)
        output = self.gamma * normalized + self.beta
        
        return output


class FeedForward:
    """
    Position-wise Feed-Forward Network
    
    [From Paper 13: Attention Is All You Need]
    
    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    """
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # First layer with ReLU (GPT-1 uses GELU, but we use ReLU for simplicity)
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)
        
        # Second layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output


class TransformerDecoderBlock:
    """
    Transformer Decoder Block
    
    [From Paper 13: Attention Is All You Need]
    
    GPT-1 uses the decoder-only architecture with:
    1. Masked Multi-Head Self-Attention
    2. Add & Norm
    3. Feed-Forward
    4. Add & Norm
    """
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        # Masked multi-head attention with residual connection
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)
        
        return x


# =============================================================================
# COMPONENTS FROM PAPER 02: THE UNREASONABLE EFFECTIVENESS OF RNNS
# =============================================================================

class Vocabulary:
    """
    Character/Token vocabulary management
    
    [From Paper 02: The Unreasonable Effectiveness of RNNs]
    """
    def __init__(self, text=None, vocab_list=None):
        if vocab_list is not None:
            self.chars = vocab_list
        elif text is not None:
            self.chars = sorted(list(set(text)))
        else:
            self.chars = []
        
        self.vocab_size = len(self.chars)
        self.char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self.ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_ix[ch] for ch in text if ch in self.char_to_ix]
    
    def decode(self, indices):
        """Convert indices to text"""
        return ''.join(self.ix_to_char.get(i, '?') for i in indices)
    
    def __len__(self):
        return self.vocab_size


def sample_from_distribution(probs, temperature=1.0):
    """
    Sample from probability distribution with temperature
    
    [From Paper 02: The Unreasonable Effectiveness of RNNs]
    
    Temperature controls randomness:
    - temp > 1: more random
    - temp = 1: standard
    - temp < 1: more deterministic
    """
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / np.sum(probs)
    
    return np.random.choice(len(probs), p=probs)


# =============================================================================
# GPT-1: COMBINING TRANSFORMER (Paper 13) + TEXT GENERATION (Paper 02)
# =============================================================================

class GPT1:
    """
    GPT-1: Generative Pre-trained Transformer
    
    Architecture from "Improving Language Understanding by Generative Pre-Training"
    (OpenAI, 2018)
    
    Components:
    - Token embeddings [Paper 02, 13]
    - Positional encoding [Paper 13]
    - Stacked Transformer decoder blocks [Paper 13]
    - Language modeling head (next-token prediction) [Paper 02, 27]
    """
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_len=512):
        """
        GPT-1 Small (original paper):
        - vocab_size: 40000 (BPE tokens)
        - d_model: 768
        - num_heads: 12
        - num_layers: 12
        - d_ff: 3072 (4 * d_model)
        - max_seq_len: 512
        - Parameters: ~117M
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Position embeddings (learned in GPT-1, not sinusoidal)
        # But we use sinusoidal from Paper 13 for educational purposes
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        
        # Transformer decoder blocks
        self.blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.final_norm = LayerNorm(d_model)
        
        # Language modeling head (predict next token)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        
        print(f"GPT-1 initialized:")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Model dimension: {d_model}")
        print(f"  - Attention heads: {num_heads}")
        print(f"  - Decoder layers: {num_layers}")
        print(f"  - Feed-forward dim: {d_ff}")
        print(f"  - Max sequence length: {max_seq_len}")
        
        # Count parameters
        total_params = (
            vocab_size * d_model +  # token embeddings
            sum([
                4 * d_model * d_model +  # attention weights (W_q, W_k, W_v, W_o)
                2 * d_model * d_ff +     # FFN weights (W1, W2)
                d_model +                # norm1 gamma, beta
                d_model +                # norm2 gamma, beta
                d_ff + d_model           # FFN biases
                for _ in range(num_layers)
            ]) +
            d_model * vocab_size        # LM head
        )
        print(f"  - Approximate parameters: {total_params:,}")
    
    def forward(self, input_ids, return_logits=True):
        """
        Forward pass through GPT-1
        
        input_ids: list or array of token indices (seq_len,)
        returns: logits for next token prediction (seq_len, vocab_size)
        """
        seq_len = len(input_ids)
        
        # Token embeddings
        x = self.token_embedding[input_ids]  # (seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:seq_len]
        
        # Create causal mask (GPT-1 uses masked self-attention)
        mask = create_causal_mask(seq_len)
        
        # Pass through transformer decoder blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final layer norm
        x = self.final_norm.forward(x)
        
        # Language modeling head
        logits = np.dot(x, self.lm_head)
        
        if return_logits:
            return logits
        else:
            return softmax(logits, axis=-1)
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0):
        """
        Generate text autoregressively
        
        [From Paper 02: sampling technique]
        [From Paper 13: transformer architecture]
        [From Paper 27: next-token prediction]
        """
        generated = list(input_ids)
        
        for _ in range(max_new_tokens):
            # Truncate if too long
            context = generated[-self.max_seq_len:]
            
            # Forward pass
            logits = self.forward(context)
            
            # Get logits for last position
            next_logits = logits[-1]
            
            # Convert to probabilities
            probs = softmax(next_logits)
            
            # Sample next token
            next_token = sample_from_distribution(probs, temperature)
            
            generated.append(next_token)
        
        return generated
    
    def compute_loss(self, input_ids, target_ids):
        """
        Compute cross-entropy loss for language modeling
        
        [From Paper 02: cross-entropy loss]
        [From Paper 27: next-token prediction]
        """
        logits = self.forward(input_ids)
        
        # Cross-entropy loss
        loss = 0
        for i, target in enumerate(target_ids):
            log_probs = np.log(softmax(logits[i]) + 1e-10)
            loss -= log_probs[target]
        
        return loss / len(target_ids)


# =============================================================================
# TRAINING UTILITIES (inspired by Paper 02 and Paper 27)
# =============================================================================

class AdamOptimizer:
    """
    Adam optimizer for training
    
    Standard optimization for transformers
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep
    
    def step(self, params, grads, name):
        """Update parameters with Adam"""
        self.t += 1
        
        if name not in self.m:
            self.m[name] = np.zeros_like(params)
            self.v[name] = np.zeros_like(params)
        
        self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grads
        self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grads ** 2)
        
        m_hat = self.m[name] / (1 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1 - self.beta2 ** self.t)
        
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


def compute_gradients_numerical(model, input_ids, target_ids, eps=1e-5):
    """
    Compute gradients numerically (for verification)
    
    Slow but useful for debugging
    """
    grads = {}
    
    # Only compute for LM head for demonstration
    base_loss = model.compute_loss(input_ids, target_ids)
    
    grad_lm_head = np.zeros_like(model.lm_head)
    for i in range(model.lm_head.shape[0]):
        for j in range(model.lm_head.shape[1]):
            model.lm_head[i, j] += eps
            loss_plus = model.compute_loss(input_ids, target_ids)
            model.lm_head[i, j] -= eps
            grad_lm_head[i, j] = (loss_plus - base_loss) / eps
    
    grads['lm_head'] = grad_lm_head
    return grads


# =============================================================================
# DEMONSTRATION: GPT-1 IN ACTION
# =============================================================================

def create_training_data():
    """Create synthetic training data with patterns"""
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test",
        "deep learning is a subset of machine learning",
        "neural networks can learn complex patterns",
        "transformers have revolutionized natural language processing",
        "attention is all you need for modern nlp",
        "gpt stands for generative pre-trained transformer",
        "language models predict the next token in a sequence",
        "pre-training followed by fine-tuning is effective",
        "self-attention allows modeling long-range dependencies",
    ] * 10  # Repeat for more training data
    
    return '\n'.join(texts)


def main():
    print("=" * 70)
    print("GPT-1 Implementation from Sutskever-30 Papers")
    print("=" * 70)
    print()
    
    # Create training data
    print("1. Preparing training data...")
    text_data = create_training_data()
    print(f"   Total characters: {len(text_data)}")
    
    # Build vocabulary
    vocab = Vocabulary(text_data)
    print(f"   Vocabulary size: {vocab.vocab_size}")
    print(f"   Sample chars: {vocab.chars[:20]}...")
    print()
    
    # Initialize GPT-1 (smaller version for demonstration)
    print("2. Initializing GPT-1...")
    model = GPT1(
        vocab_size=vocab.vocab_size,
        d_model=64,        # Smaller for demo (original: 768)
        num_heads=4,       # Smaller for demo (original: 12)
        num_layers=2,      # Smaller for demo (original: 12)
        d_ff=256,          # Smaller for demo (original: 3072)
        max_seq_len=128    # Smaller for demo (original: 512)
    )
    print()
    
    # Test forward pass
    print("3. Testing forward pass...")
    test_text = "the quick brown"
    test_ids = vocab.encode(test_text)
    print(f"   Input: '{test_text}'")
    print(f"   Token IDs: {test_ids}")
    
    logits = model.forward(test_ids)
    print(f"   Output logits shape: {logits.shape}")
    print(f"   Expected: ({len(test_ids)}, {vocab.vocab_size})")
    print()
    
    # Test generation
    print("4. Testing text generation (before training)...")
    seed_text = "the"
    seed_ids = vocab.encode(seed_text)
    generated_ids = model.generate(seed_ids, max_new_tokens=30, temperature=0.8)
    generated_text = vocab.decode(generated_ids)
    print(f"   Seed: '{seed_text}'")
    print(f"   Generated: '{generated_text}'")
    print("   (Random output expected before training)")
    print()
    
    # Demonstrate attention patterns
    print("5. Visualizing attention patterns...")
    visualize_attention(model, test_ids, vocab)
    print()
    
    # Training loop (simplified)
    print("6. Running simplified training loop...")
    train_model(model, vocab, text_data, num_iterations=200)
    print()
    
    # Test generation after training
    print("7. Testing text generation (after training)...")
    generated_ids = model.generate(seed_ids, max_new_tokens=30, temperature=0.8)
    generated_text = vocab.decode(generated_ids)
    print(f"   Seed: '{seed_text}'")
    print(f"   Generated: '{generated_text}'")
    print()
    
    # Loss comparison
    print("8. Final loss evaluation...")
    final_loss = model.compute_loss(test_ids, test_ids[1:] + [test_ids[0]])
    print(f"   Test loss: {final_loss:.4f}")
    print()
    
    print("=" * 70)
    print("GPT-1 Components from Sutskever-30 Papers:")
    print("=" * 70)
    print()
    print("From Paper 02 (The Unreasonable Effectiveness of RNNs):")
    print("  - Vocabulary management and tokenization")
    print("  - Sampling from probability distributions")
    print("  - Cross-entropy loss for language modeling")
    print("  - Autoregressive text generation")
    print()
    print("From Paper 13 (Attention Is All You Need):")
    print("  - Scaled dot-product attention mechanism")
    print("  - Multi-head attention architecture")
    print("  - Positional encoding for sequence position")
    print("  - Layer normalization")
    print("  - Feed-forward networks")
    print("  - Residual connections")
    print("  - Causal masking for autoregressive generation")
    print()
    print("From Paper 27 (Multi-token Prediction):")
    print("  - Next-token prediction objective")
    print("  - Language modeling head architecture")
    print("  - Training signal from sequence prediction")
    print()
    print("=" * 70)
    print("GPT-1 Architecture Summary:")
    print("=" * 70)
    print()
    print("Input: Token IDs -> Token Embeddings + Positional Encodings")
    print("   |")
    print("   v")
    print("For each Transformer Decoder Block:")
    print("   |")
    print("   +-- Masked Multi-Head Self-Attention")
    print("   |   (Can only attend to current and past positions)")
    print("   |")
    print("   +-- Add & Layer Norm (residual connection)")
    print("   |")
    print("   +-- Feed-Forward Network")
    print("   |")
    print("   +-- Add & Layer Norm (residual connection)")
    print("   |")
    print("   v")
    print("Final Layer Norm")
    print("   |")
    print("   v")
    print("Language Modeling Head -> Predict next token probabilities")
    print()


def visualize_attention(model, input_ids, vocab):
    """Visualize attention patterns from the first transformer block"""
    # Get the first block's attention
    block = model.blocks[0]
    
    # Forward pass to populate attention weights
    _ = model.forward(input_ids)
    
    # Get attention weights
    attn_weights = block.attention.attention_weights
    
    # Plot
    fig, axes = plt.subplots(1, min(4, model.num_heads), figsize=(16, 4))
    if model.num_heads == 1:
        axes = [axes]
    
    tokens = [vocab.decode([tid]) for tid in input_ids]
    
    for i, ax in enumerate(axes[:model.num_heads]):
        im = ax.imshow(attn_weights[i], cmap='viridis', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {i+1}')
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.colorbar(im, ax=axes, label='Attention Weight', fraction=0.046, pad=0.04)
    plt.suptitle('GPT-1 Multi-Head Attention Patterns (First Block)', fontsize=14)
    plt.tight_layout()
    plt.savefig('/home/z/my-project/download/gpt1_attention_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Attention visualization saved to: gpt1_attention_patterns.png")


def train_model(model, vocab, text_data, num_iterations=200):
    """Simplified training loop demonstration"""
    # Prepare sequences
    seq_length = 20
    encoded_data = vocab.encode(text_data)
    
    losses = []
    learning_rate = 0.001
    
    print("   Training (simplified - updating LM head only)...")
    
    for iteration in range(num_iterations):
        # Random starting position
        start = np.random.randint(0, len(encoded_data) - seq_length - 1)
        
        # Input and target sequences
        input_ids = encoded_data[start:start + seq_length]
        target_ids = encoded_data[start + 1:start + seq_length + 1]
        
        # Forward pass
        logits = model.forward(input_ids)
        
        # Compute loss
        loss = model.compute_loss(input_ids, target_ids)
        losses.append(loss)
        
        # Simple gradient descent on LM head (simplified training)
        if iteration < num_iterations - 1:
            # Compute output gradient
            probs = softmax(logits, axis=-1)
            grad = probs.copy()
            for i, target in enumerate(target_ids):
                grad[i, target] -= 1
            grad /= len(target_ids)
            
            # Update LM head (simplified - just LM head)
            hidden = model.final_norm.forward(
                model.blocks[-1].forward(
                    model.token_embedding[input_ids] + model.pos_encoding[:seq_length],
                    create_causal_mask(seq_length)
                )
            )
            
            # Gradient for LM head
            grad_lm_head = np.dot(hidden.T, grad)
            model.lm_head -= learning_rate * grad_lm_head
        
        if (iteration + 1) % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            print(f"   Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses, linewidth=1)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('GPT-1 Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/z/my-project/download/gpt1_training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Training loss plot saved to: gpt1_training_loss.png")


if __name__ == "__main__":
    main()
