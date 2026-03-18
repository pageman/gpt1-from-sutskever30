"""
GPT-1 Complete Implementation with Full Backpropagation
=======================================================

This is a complete implementation of GPT-1 with manual backpropagation
for all components, inspired by the Sutskever-30 repository's approach
of implementing everything from scratch in NumPy.

Components adapted from:
- Paper 02: The Unreasonable Effectiveness of RNNs (text generation, training loop)
- Paper 03: Understanding LSTM Networks (gradient flow concepts)
- Paper 13: Attention Is All You Need (complete transformer with backprop)
- Paper 27: Multi-token Prediction (next-token prediction objective)
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

np.random.seed(42)

# =============================================================================
# UTILITIES
# =============================================================================

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x):
    """
    Gaussian Error Linear Unit (used in GPT-1/2/3)
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
    Approximated as: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(x):
    """Backward pass for GELU activation"""
    cdf = 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)
    return cdf + x * pdf * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3))) * (
        1 - np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)) ** 2
    ) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x ** 2)


# =============================================================================
# LAYER NORMALIZATION WITH BACKPROP
# =============================================================================

class LayerNorm:
    """Layer Normalization with backward pass"""
    
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.eps = eps
        
        # Cache for backward
        self.cache = {}
        
        # Gradients
        self.dgamma = None
        self.dbeta = None
    
    def forward(self, x):
        """Forward pass"""
        self.cache['x'] = x
        self.cache['mean'] = x.mean(axis=-1, keepdims=True)
        self.cache['std'] = x.std(axis=-1, keepdims=True)
        self.cache['normalized'] = (x - self.cache['mean']) / (self.cache['std'] + self.eps)
        
        return self.gamma * self.cache['normalized'] + self.beta
    
    def backward(self, dout):
        """Backward pass"""
        x = self.cache['x']
        normalized = self.cache['normalized']
        mean = self.cache['mean']
        std = self.cache['std']
        N = x.shape[-1]
        
        # Gradients for gamma and beta
        self.dgamma = np.sum(dout * normalized, axis=tuple(range(dout.ndim - 1)))
        self.dbeta = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
        
        # Gradient for normalized
        dnormalized = dout * self.gamma
        
        # Gradient for x (chain rule through normalization)
        dx_norm = dnormalized
        dx_var = np.sum(dx_norm * (x - mean) * -0.5 * (std + self.eps) ** -3, axis=-1, keepdims=True)
        dx_mean = np.sum(dx_norm * -1 / (std + self.eps), axis=-1, keepdims=True) + dx_var * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        
        dx = dx_norm / (std + self.eps) + dx_var * 2 * (x - mean) / N + dx_mean / N
        
        return dx
    
    def get_params(self):
        return {'gamma': self.gamma, 'beta': self.beta}
    
    def get_grads(self):
        return {'gamma': self.dgamma, 'beta': self.dbeta}
    
    def set_params(self, params):
        self.gamma = params['gamma']
        self.beta = params['beta']


# =============================================================================
# MULTI-HEAD ATTENTION WITH BACKPROP
# =============================================================================

class MultiHeadAttention:
    """Multi-Head Attention with complete backward pass"""
    
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights with Xavier initialization
        scale = np.sqrt(2.0 / (d_model + d_model))
        self.W_q = np.random.randn(d_model, d_model) * scale
        self.W_k = np.random.randn(d_model, d_model) * scale
        self.W_v = np.random.randn(d_model, d_model) * scale
        self.W_o = np.random.randn(d_model, d_model) * scale
        
        # Cache for backward
        self.cache = {}
        
        # Gradients
        self.dW_q = None
        self.dW_k = None
        self.dW_v = None
        self.dW_o = None
    
    def split_heads(self, x):
        """Split into multiple heads"""
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)  # (num_heads, seq_len, d_k)
    
    def combine_heads(self, x):
        """Combine heads"""
        seq_len = x.shape[1]
        x = x.transpose(1, 0, 2)
        return x.reshape(seq_len, self.d_model)
    
    def forward(self, Q, K, V, mask=None):
        """Forward pass with caching for backward"""
        self.cache['Q_input'] = Q
        self.cache['K_input'] = K
        self.cache['V_input'] = V
        self.cache['mask'] = mask
        
        # Linear projections
        Q_proj = np.dot(Q, self.W_q.T)
        K_proj = np.dot(K, self.W_k.T)
        V_proj = np.dot(V, self.W_v.T)
        
        self.cache['Q_proj'] = Q_proj
        self.cache['K_proj'] = K_proj
        self.cache['V_proj'] = V_proj
        
        # Split heads
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)
        
        # Store for backward
        self.cache['Q_heads'] = Q_heads
        self.cache['K_heads'] = K_heads
        self.cache['V_heads'] = V_heads
        
        # Attention for each head
        head_outputs = []
        attention_weights = []
        
        for i in range(self.num_heads):
            # Scaled dot-product attention
            scores = np.dot(Q_heads[i], K_heads[i].T) / np.sqrt(self.d_k)
            
            if mask is not None:
                scores = scores + (mask * -1e9)
            
            attn = softmax(scores, axis=-1)
            attention_weights.append(attn)
            head_outputs.append(np.dot(attn, V_heads[i]))
        
        self.cache['attention_weights'] = attention_weights
        
        # Combine heads
        heads = np.stack(head_outputs, axis=0)
        combined = self.combine_heads(heads)
        
        # Output projection
        output = np.dot(combined, self.W_o.T)
        
        return output
    
    def backward(self, dout):
        """Complete backward pass for multi-head attention"""
        Q = self.cache['Q_input']
        K = self.cache['K_input']
        V = self.cache['V_input']
        mask = self.cache['mask']
        Q_heads = self.cache['Q_heads']
        K_heads = self.cache['K_heads']
        V_heads = self.cache['V_heads']
        attn_weights = self.cache['attention_weights']
        
        # Gradient through output projection
        dcombined = np.dot(dout, self.W_o)
        self.dW_o = np.dot(dout.T, self.combine_heads(
            np.stack([np.dot(attn_weights[i], V_heads[i]) for i in range(self.num_heads)], axis=0)
        ))
        
        # Split gradient for heads
        dheads = self.split_heads(dcombined)
        
        # Gradients for each head
        dQ_heads = np.zeros_like(Q_heads)
        dK_heads = np.zeros_like(K_heads)
        dV_heads = np.zeros_like(V_heads)
        
        for i in range(self.num_heads):
            attn = attn_weights[i]
            
            # Gradient through attention output
            dV_heads[i] = np.dot(attn.T, dheads[i])
            
            # Gradient through attention weights
            dattn = np.dot(dheads[i], V_heads[i].T)
            
            # Gradient through softmax
            dscores = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
            dscores = dscores / np.sqrt(self.d_k)
            
            # Gradient through Q and K
            dQ_heads[i] = np.dot(dscores, K_heads[i])
            dK_heads[i] = np.dot(dscores.T, Q_heads[i])
        
        # Combine head gradients
        dQ_proj = self.combine_heads(dQ_heads)
        dK_proj = self.combine_heads(dK_heads)
        dV_proj = self.combine_heads(dV_heads)
        
        # Gradient through projection weights
        self.dW_q = np.dot(dQ_proj.T, Q)
        self.dW_k = np.dot(dK_proj.T, K)
        self.dW_v = np.dot(dV_proj.T, V)
        
        # Gradient for inputs
        dQ = np.dot(dQ_proj, self.W_q)
        dK = np.dot(dK_proj, self.W_k)
        dV = np.dot(dV_proj, self.W_v)
        
        return dQ, dK, dV
    
    def get_params(self):
        return {'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v, 'W_o': self.W_o}
    
    def get_grads(self):
        return {'W_q': self.dW_q, 'W_k': self.dW_k, 'W_v': self.dW_v, 'W_o': self.dW_o}


# =============================================================================
# FEED-FORWARD NETWORK WITH BACKPROP
# =============================================================================

class FeedForward:
    """Position-wise Feed-Forward Network with backward pass"""
    
    def __init__(self, d_model, d_ff):
        # Xavier initialization
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / (d_model + d_ff))
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / (d_ff + d_model))
        self.b2 = np.zeros(d_model)
        
        self.cache = {}
        
        # Gradients
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None
    
    def forward(self, x):
        """Forward pass with GELU activation"""
        self.cache['x'] = x
        
        # First layer
        hidden = np.dot(x, self.W1) + self.b1
        self.cache['hidden_pre_act'] = hidden
        
        # GELU activation
        hidden = gelu(hidden)
        self.cache['hidden'] = hidden
        
        # Second layer
        output = np.dot(hidden, self.W2) + self.b2
        
        return output
    
    def backward(self, dout):
        """Backward pass"""
        x = self.cache['x']
        hidden = self.cache['hidden']
        hidden_pre = self.cache['hidden_pre_act']
        
        # Gradient through second layer
        self.dW2 = np.dot(hidden.T, dout)
        self.db2 = np.sum(dout, axis=0)
        dhidden = np.dot(dout, self.W2.T)
        
        # Gradient through GELU
        dhidden = dhidden * gelu_backward(hidden_pre)
        
        # Gradient through first layer
        self.dW1 = np.dot(x.T, dhidden)
        self.db1 = np.sum(dhidden, axis=0)
        
        # Gradient for input
        dx = np.dot(dhidden, self.W1.T)
        
        return dx
    
    def get_params(self):
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
    
    def get_grads(self):
        return {'W1': self.dW1, 'b1': self.db1, 'W2': self.dW2, 'b2': self.db2}


# =============================================================================
# TRANSFORMER DECODER BLOCK
# =============================================================================

class TransformerDecoderBlock:
    """Transformer Decoder Block with backward pass"""
    
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        
        self.cache = {}
    
    def forward(self, x, mask=None):
        """Forward pass"""
        self.cache['x1'] = x
        
        # Self-attention with residual
        attn_out = self.attention.forward(x, x, x, mask)
        self.cache['attn_out'] = attn_out
        x = self.norm1.forward(x + attn_out)
        
        self.cache['x2'] = x
        
        # Feed-forward with residual
        ff_out = self.ff.forward(x)
        self.cache['ff_out'] = ff_out
        x = self.norm2.forward(x + ff_out)
        
        return x
    
    def backward(self, dout):
        """Backward pass"""
        # Gradient through norm2
        dnorm2 = self.norm2.backward(dout)
        
        # Residual connection gradient
        dx2 = dnorm2 + self.ff.backward(dnorm2)
        
        # Gradient through norm1
        dnorm1 = self.norm1.backward(dx2)
        
        # Self-attention backward
        dQ, dK, dV = self.attention.backward(dnorm1)
        
        # Residual connection gradient
        dx1 = dnorm1 + dQ + dK + dV
        
        return dx1
    
    def get_params(self):
        params = {}
        params['attention'] = self.attention.get_params()
        params['norm1'] = self.norm1.get_params()
        params['ff'] = self.ff.get_params()
        params['norm2'] = self.norm2.get_params()
        return params
    
    def get_grads(self):
        grads = {}
        grads['attention'] = self.attention.get_grads()
        grads['norm1'] = self.norm1.get_grads()
        grads['ff'] = self.ff.get_grads()
        grads['norm2'] = self.norm2.get_grads()
        return grads


# =============================================================================
# GPT-1 COMPLETE MODEL
# =============================================================================

class GPT1:
    """
    Complete GPT-1 implementation with full backpropagation
    
    This combines components from multiple Sutskever-30 papers:
    - Paper 02: Text generation and vocabulary
    - Paper 13: Transformer architecture
    - Paper 27: Next-token prediction
    """
    
    def __init__(self, vocab_size, d_model=768, num_heads=12, num_layers=12, 
                 d_ff=3072, max_seq_len=512):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        
        # Position embeddings (learned in GPT)
        self.position_embedding = np.random.randn(max_seq_len, d_model) * 0.02
        
        # Transformer blocks
        self.blocks = [
            TransformerDecoderBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Final layer norm
        self.final_norm = LayerNorm(d_model)
        
        # Language modeling head (tied with embeddings in GPT-2, but separate here)
        self.lm_head = np.random.randn(d_model, vocab_size) * 0.02
        
        self.cache = {}
        
        # Count parameters
        self._count_parameters()
    
    def _count_parameters(self):
        """Count total parameters"""
        total = 0
        total += self.vocab_size * self.d_model  # Token embeddings
        total += self.max_seq_len * self.d_model  # Position embeddings
        
        for block in self.blocks:
            params = block.get_params()
            # Attention: 4 weight matrices
            total += 4 * self.d_model * self.d_model
            # FFN: 2 weight matrices + 2 biases
            total += 2 * self.d_model * self.d_ff + self.d_ff + self.d_model
            # Layer norms: 2 * (gamma + beta)
            total += 4 * self.d_model
        
        total += 2 * self.d_model  # Final layer norm
        total += self.d_model * self.vocab_size  # LM head
        
        print(f"GPT-1 Model:")
        print(f"  Parameters: {total:,}")
        print(f"  Vocabulary: {self.vocab_size}")
        print(f"  Model dim: {self.d_model}")
        print(f"  Heads: {self.num_heads}")
        print(f"  Layers: {self.num_layers}")
        print(f"  FFN dim: {self.d_ff}")
        print(f"  Max seq len: {self.max_seq_len}")
    
    def forward(self, input_ids):
        """Forward pass through GPT-1"""
        seq_len = len(input_ids)
        self.cache['input_ids'] = input_ids
        self.cache['seq_len'] = seq_len
        
        # Token embeddings
        x = self.token_embedding[input_ids]
        self.cache['token_emb'] = x.copy()
        
        # Position embeddings
        pos_emb = self.position_embedding[:seq_len]
        self.cache['pos_emb'] = pos_emb
        x = x + pos_emb
        
        # Causal mask
        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        
        # Transformer blocks
        for block in self.blocks:
            x = block.forward(x, mask)
        
        # Final layer norm
        x = self.final_norm.forward(x)
        self.cache['hidden'] = x.copy()
        
        # Language modeling head
        logits = np.dot(x, self.lm_head)
        
        return logits
    
    def backward(self, dlogits):
        """Backward pass through entire model"""
        # Gradient through LM head
        self.dlm_head = np.dot(self.cache['hidden'].T, dlogits)
        dhidden = np.dot(dlogits, self.lm_head.T)
        
        # Gradient through final norm
        dx = self.final_norm.backward(dhidden)
        
        # Gradient through transformer blocks (reverse order)
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        
        # Gradient through embeddings
        self.dtoken_embedding = np.zeros_like(self.token_embedding)
        for i, idx in enumerate(self.cache['input_ids']):
            self.dtoken_embedding[idx] += dx[i]
        
        self.dposition_embedding = np.zeros_like(self.position_embedding)
        self.dposition_embedding[:self.cache['seq_len']] = dx
        
        return dx
    
    def compute_loss(self, input_ids, target_ids):
        """Compute cross-entropy loss with caching for backward"""
        logits = self.forward(input_ids)
        self.cache['logits'] = logits
        self.cache['target_ids'] = target_ids
        
        # Softmax
        probs = softmax(logits, axis=-1)
        self.cache['probs'] = probs
        
        # Cross-entropy loss
        loss = 0
        for i, target in enumerate(target_ids):
            loss -= np.log(probs[i, target] + 1e-10)
        
        return loss / len(target_ids)
    
    def backward_from_loss(self):
        """Compute gradient from loss"""
        probs = self.cache['probs']
        target_ids = self.cache['target_ids']
        
        # Gradient of cross-entropy with softmax
        dlogits = probs.copy()
        for i, target in enumerate(target_ids):
            dlogits[i, target] -= 1
        dlogits /= len(target_ids)
        
        return dlogits
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """Generate text autoregressively"""
        generated = list(input_ids)
        
        for _ in range(max_new_tokens):
            # Truncate to max length
            context = generated[-self.max_seq_len:]
            
            # Forward pass
            logits = self.forward(context)
            
            # Get last position logits
            next_logits = logits[-1]
            
            # Apply temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Top-k sampling
            if top_k is not None:
                indices = np.argsort(next_logits)[-top_k:]
                next_logits = np.array([next_logits[i] for i in indices])
                probs = softmax(next_logits)
                next_idx = np.random.choice(indices, p=probs)
            else:
                probs = softmax(next_logits)
                next_idx = np.random.choice(len(probs), p=probs)
            
            generated.append(next_idx)
        
        return generated
    
    def get_all_params(self):
        """Get all parameters"""
        params = {
            'token_embedding': self.token_embedding,
            'position_embedding': self.position_embedding,
            'lm_head': self.lm_head,
        }
        for i, block in enumerate(self.blocks):
            params[f'block_{i}'] = block.get_params()
        params['final_norm'] = self.final_norm.get_params()
        return params
    
    def get_all_grads(self):
        """Get all gradients"""
        grads = {
            'token_embedding': self.dtoken_embedding,
            'position_embedding': self.dposition_embedding,
            'lm_head': self.dlm_head,
        }
        for i, block in enumerate(self.blocks):
            grads[f'block_{i}'] = block.get_grads()
        grads['final_norm'] = self.final_norm.get_grads()
        return grads


# =============================================================================
# ADAM OPTIMIZER
# =============================================================================

class AdamOptimizer:
    """Adam optimizer with weight decay"""
    
    def __init__(self, lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0
    
    def step(self, params, grads):
        """Update parameters with Adam - modifies params in place"""
        self.t += 1
        
        for name in params:
            if name not in grads:
                continue
                
            if name not in self.m:
                self.m[name] = np.zeros_like(params[name])
                self.v[name] = np.zeros_like(params[name])
            
            g = grads[name]
            if self.weight_decay > 0:
                g = g + self.weight_decay * params[name]
            
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * g
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)
            
            # Update in place
            params[name] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def create_bpe_vocabulary(text, num_merges=100):
    """Simple BPE-like vocabulary creation"""
    # Start with character vocabulary
    vocab = list(set(text))
    vocab.extend(['<pad>', '<unk>', '<eos>'])
    
    # Count word frequencies
    words = text.split()
    word_freqs = Counter(words)
    
    # Simple merge operations
    for _ in range(min(num_merges, len(vocab) - 3)):
        # Find most common pair
        pairs = Counter()
        for word, freq in word_freqs.items():
            chars = list(word)
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i+1])] += freq
        
        if not pairs:
            break
        
        best_pair = pairs.most_common(1)[0][0]
        
        # Add merged token to vocabulary
        new_token = best_pair[0] + best_pair[1]
        if new_token not in vocab:
            vocab.append(new_token)
    
    return vocab


def encode_text(text, vocab):
    """Encode text using vocabulary"""
    char_to_ix = {ch: i for i, ch in enumerate(vocab)}
    return [char_to_ix.get(ch, vocab.index('<unk>')) for ch in text]


def decode_tokens(indices, vocab):
    """Decode token indices to text"""
    return ''.join(vocab[i] if i < len(vocab) else '<unk>' for i in indices)


def flatten_params(params, prefix=''):
    """Recursively flatten nested parameter dictionaries"""
    flat = {}
    for name, value in params.items():
        full_name = f"{prefix}_{name}" if prefix else name
        if isinstance(value, dict):
            flat.update(flatten_params(value, full_name))
        else:
            flat[full_name] = value
    return flat


def train_gpt1(model, train_data, vocab, num_epochs=10, batch_size=4, 
               seq_length=32, lr=1e-4, print_every=100):
    """Train GPT-1 model"""
    optimizer = AdamOptimizer(lr=lr, weight_decay=0.01)
    losses = []
    
    # Prepare data
    encoded = encode_text(train_data, vocab)
    
    num_batches = min(500, (len(encoded) - seq_length) // batch_size)  # Limit batches
    
    print(f"\nTraining:")
    print(f"  Data size: {len(encoded)} tokens")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_length}")
    print(f"  Batches per epoch: {num_batches}")
    print()
    
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch in range(num_batches):
            # Get batch
            batch_loss = 0
            
            for b in range(batch_size):
                start = batch * batch_size + b
                start = start % (len(encoded) - seq_length - 1)
                
                input_ids = encoded[start:start + seq_length]
                target_ids = encoded[start + 1:start + seq_length + 1]
                
                # Forward pass
                loss = model.compute_loss(input_ids, target_ids)
                batch_loss += loss
                
                # Backward pass
                dlogits = model.backward_from_loss()
                model.backward(dlogits)
            
            batch_loss /= batch_size
            epoch_loss += batch_loss
            losses.append(batch_loss)
            
            # Update parameters - flatten properly
            all_params = model.get_all_params()
            all_grads = model.get_all_grads()
            
            flat_params = flatten_params(all_params)
            flat_grads = flatten_params(all_grads)
            
            # Filter out None grads
            flat_grads = {k: v for k, v in flat_grads.items() if v is not None}
            
            optimizer.step(flat_params, flat_grads)
            
            if (batch + 1) % print_every == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch+1}, Batch {batch+1}/{num_batches}, "
                      f"Loss: {batch_loss:.4f}, Time: {elapsed:.1f}s")
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
        
        # Generate sample
        seed = encode_text("the ", vocab)
        generated = model.generate(seed, max_new_tokens=20, temperature=0.8)
        print(f"Sample: '{decode_tokens(generated, vocab)}'")
        print()
    
    return losses


# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    print("=" * 70)
    print("GPT-1 Complete Implementation with Backpropagation")
    print("From Sutskever-30 Papers")
    print("=" * 70)
    print()
    
    # Create training data
    print("1. Creating training data...")
    training_text = """
    The quick brown fox jumps over the lazy dog.
    Deep learning is a subset of machine learning.
    Neural networks learn complex patterns from data.
    Transformers have revolutionized natural language processing.
    Attention mechanisms allow models to focus on relevant information.
    GPT stands for Generative Pre-trained Transformer.
    Language models predict the next token in a sequence.
    Self-attention enables modeling long-range dependencies.
    The decoder-only architecture is used for text generation.
    Training large language models requires significant compute.
    """ * 50
    
    # Build vocabulary
    vocab = sorted(list(set(training_text)))
    vocab.extend(['<pad>', '<unk>', '<eos>'])
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Training text length: {len(training_text)} characters")
    print()
    
    # Initialize model (smaller for demonstration)
    print("2. Initializing GPT-1 model...")
    model = GPT1(
        vocab_size=len(vocab),
        d_model=128,
        num_heads=4,
        num_layers=2,
        d_ff=512,
        max_seq_len=256
    )
    print()
    
    # Test forward pass
    print("3. Testing forward pass...")
    test_input = encode_text("hello world", vocab)[:10]
    logits = model.forward(test_input)
    print(f"   Input length: {len(test_input)}")
    print(f"   Output shape: {logits.shape}")
    print()
    
    # Test generation before training
    print("4. Testing generation (before training)...")
    seed = encode_text("the", vocab)
    generated = model.generate(seed, max_new_tokens=30, temperature=0.8)
    print(f"   Generated: '{decode_tokens(generated, vocab)}'")
    print()
    
    # Train model
    print("5. Training model...")
    losses = train_gpt1(
        model, training_text, vocab,
        num_epochs=3,
        batch_size=2,
        seq_length=32,
        lr=1e-3,
        print_every=50
    )
    
    # Test generation after training
    print("6. Testing generation (after training)...")
    for temp in [0.5, 0.8, 1.0]:
        generated = model.generate(seed, max_new_tokens=30, temperature=temp)
        print(f"   Temperature {temp}: '{decode_tokens(generated, vocab)}'")
    print()
    
    # Plot training loss
    plt.figure(figsize=(12, 5))
    plt.plot(losses, linewidth=1)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('GPT-1 Training Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/z/my-project/download/gpt1_complete_training.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("7. Training plot saved to gpt1_complete_training.png")
    
    print()
    print("=" * 70)
    print("Implementation Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
