# GPT-1 Implementation from Sutskever-30 Papers

## Overview

This document describes the construction of GPT-1 ("Improving Language Understanding by Generative Pre-Training", OpenAI 2018) using implementations from the Sutskever-30 papers repository at https://github.com/pageman/sutskever-30-implementations/

GPT-1 is fundamentally a **Transformer Decoder** architecture with:
1. Token embeddings + Positional embeddings
2. Stacked Transformer decoder blocks (masked self-attention + feed-forward)
3. Language modeling head for next-token prediction
4. Pre-training objective: maximize likelihood of next token

## Papers Used

### Paper 02: The Unreasonable Effectiveness of RNNs (Andrej Karpathy)
**Components extracted:**
- Vocabulary management and character-level tokenization
- Sampling from probability distributions (temperature-based)
- Cross-entropy loss for language modeling
- Autoregressive text generation loop
- Training loop structure with gradient updates

**Code adapted:**
```python
# Vocabulary building
chars = sorted(list(set(data)))
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}

# Sampling with temperature
def sample_from_distribution(probs, temperature=1.0):
    if temperature != 1.0:
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / np.sum(probs)
    return np.random.choice(len(probs), p=probs)
```

### Paper 13: Attention Is All You Need (Vaswani et al.)
**Components extracted:**
- Scaled dot-product attention mechanism
- Multi-head attention with parallel computation
- Positional encoding (sinusoidal)
- Layer normalization
- Feed-forward networks with residual connections
- Causal masking for autoregressive generation

**Core architecture code:**
```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = scores + (mask * -1e9)
    attention_weights = softmax(scores, axis=-1)
    output = np.dot(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention:
    def forward(self, Q, K, V, mask=None):
        # Linear projections for Q, K, V
        # Split into multiple heads
        # Apply attention per head
        # Combine and project output
        ...

class TransformerDecoderBlock:
    def forward(self, x, mask=None):
        # Masked multi-head attention + residual + norm
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)
        # Feed-forward + residual + norm
        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)
        return x
```

### Paper 27: Multi-Token Prediction (Meta AI)
**Components extracted:**
- Next-token prediction objective
- Language modeling head architecture
- Training signal from sequence prediction
- Sample efficiency concepts

**Concept applied:**
GPT-1 uses single-token prediction at each position:
```
Given: [w1, w2, w3, w4]
Predict: w5 at each position
Loss: -sum(log P(target_i | context_i))
```

## GPT-1 Architecture

```
Input: Token IDs
   |
   v
Token Embedding + Position Embedding
   |
   v
+---------------------------+
| Transformer Decoder Block |
|                           |
|  +--------------------+   |
|  | Masked Multi-Head  |   |
|  | Self-Attention     |   |
|  +--------------------+   |
|           |               |
|    Add & Layer Norm       |
|           |               |
|  +--------------------+   |
|  | Feed-Forward Network|  |
|  +--------------------+   |
|           |               |
|    Add & Layer Norm       |
+---------------------------+
   | (repeated N times)
   v
Final Layer Norm
   |
   v
Language Modeling Head
   |
   v
Token Probabilities
```

## Implementation Files

### gpt1_from_sutskever30.py
- **Purpose**: Educational implementation showing component mapping
- **Features**:
  - Complete GPT-1 architecture
  - Forward pass implementation
  - Simple text generation
  - Attention pattern visualization
  - Simplified training (LM head only)

### gpt1_complete_implementation.py
- **Purpose**: Full implementation with complete backpropagation
- **Features**:
  - Complete backward pass for all layers
  - Layer norm backward
  - Multi-head attention backward
  - Feed-forward backward (with GELU)
  - Adam optimizer with weight decay
  - Full training loop
  - Temperature-based sampling

## Key Components in Detail

### 1. Scaled Dot-Product Attention
From Paper 13:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The scaling factor $\frac{1}{\sqrt{d_k}}$ prevents the dot products from growing too large.

### 2. Multi-Head Attention
From Paper 13:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1, ..., head_h)W^O$$

Each head can learn different attention patterns (e.g., syntactic, semantic).

### 3. Positional Encoding
From Paper 13:
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

GPT-1 uses learned position embeddings instead, but sinusoidal is shown for educational purposes.

### 4. Causal Masking
Critical for autoregressive generation - prevents attending to future tokens:
```python
mask = np.triu(np.ones((seq_len, seq_len)), k=1)
```

### 5. Layer Normalization
Normalizes across features (not batch):
```python
def forward(self, x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return self.gamma * (x - mean) / (std + eps) + self.beta
```

## Training

The training loop follows the pattern from Paper 02 (Char-RNN):

1. Sample random sequences from training data
2. Forward pass through model
3. Compute cross-entropy loss
4. Backward pass (backpropagation)
5. Update parameters with Adam optimizer

**Loss function:**
$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})$$

## Generated Visualizations

- `gpt1_attention_patterns.png`: Multi-head attention weight visualization
- `gpt1_training_loss.png`: Training loss curve
- `gpt1_complete_training.png`: Complete training loss curve

## Running the Code

```bash
# Simple demonstration
python3 gpt1_from_sutskever30.py

# Complete implementation with backpropagation
python3 gpt1_complete_implementation.py
```

## Model Configuration (Original GPT-1)

| Parameter | GPT-1 (Original) | Demo Version |
|-----------|-----------------|--------------|
| vocab_size | 40,000 | ~50 |
| d_model | 768 | 64-128 |
| num_heads | 12 | 4 |
| num_layers | 12 | 2 |
| d_ff | 3,072 | 256-512 |
| max_seq_len | 512 | 128-256 |
| Parameters | ~117M | ~100K-400K |

## Key Takeaways

1. **GPT-1 = Transformer Decoder**: The architecture is a stack of transformer decoder blocks with masked self-attention.

2. **From Paper 02**: Text generation concepts - vocabulary, sampling, autoregressive loop.

3. **From Paper 13**: The core transformer machinery - attention, normalization, feed-forward networks.

4. **From Paper 27**: The prediction objective - next-token prediction provides the training signal.

5. **NumPy-only**: Like the Sutskever-30 repository, all operations are implemented from scratch for educational clarity.

## References

1. Radford, A., et al. (2018). "Improving Language Understanding by Generative Pre-Training"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. Karpathy, A. (2015). "The Unreasonable Effectiveness of Recurrent Neural Networks"
4. Gloeckle, F., et al. (2024). "Better & Faster Large Language Models via Multi-token Prediction"
5. https://github.com/pageman/sutskever-30-implementations/
