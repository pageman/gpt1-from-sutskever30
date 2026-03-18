# GPT-1 from the Sutskever 30

> GPT-1 built entirely from scratch in NumPy, assembled from three papers on Ilya Sutskever's foundational reading list.

No PyTorch. No TensorFlow. Every forward pass, every gradient, every optimizer step — written by hand.

---

## The Idea

Ilya Sutskever's 30-paper reading list is widely regarded as the essential curriculum for understanding modern deep learning. The companion repository [sutskever-30-implementations](https://github.com/pageman/sutskever-30-implementations) implements each paper individually in NumPy. This project goes one step further: it assembles three of those papers into a working GPT-1.

GPT-1 is not a novel architecture — it is a synthesis. This implementation makes that lineage explicit.

---

## Paper Lineage

| Paper | Authors | Contribution to GPT-1 |
|-------|---------|----------------------|
| **Paper 02** | Karpathy — *The Unreasonable Effectiveness of RNNs* | Vocabulary management, character tokenization, cross-entropy loss, autoregressive generation loop, temperature sampling |
| **Paper 13** | Vaswani et al. — *Attention Is All You Need* | Scaled dot-product attention, multi-head attention, positional encoding, layer normalization, feed-forward blocks, residual connections, causal masking |
| **Paper 27** | Gloeckle et al. — *Better & Faster LLMs via Multi-Token Prediction* | Next-token prediction objective, language modeling head architecture, training signal |

Three papers. One language model. That's the whole story.

---

## Architecture

```
Input Token IDs
      |
      v
Token Embedding + Learned Position Embedding
      |
      v
+-----------------------------+
|  Transformer Decoder Block  |  x N layers
|                             |
|  Masked Multi-Head          |
|  Self-Attention             |
|       +                     |
|  Add & Layer Norm           |
|       +                     |
|  Feed-Forward (GELU)        |
|       +                     |
|  Add & Layer Norm           |
+-----------------------------+
      |
      v
Final Layer Norm
      |
      v
Language Modeling Head
      |
      v
Next-Token Probabilities
```

---

## What's Implemented

### `gpt1_from_sutskever30.py` — Educational walkthrough
- Full forward pass with inline paper citations
- Causal masking, sinusoidal positional encoding
- Autoregressive text generation with temperature
- Multi-head attention pattern visualization (matplotlib)

### `gpt1_complete_implementation.py` — Full training from scratch
- **Complete backpropagation** through every layer: attention, feed-forward, layer norm, embeddings
- **GELU activation** with analytic backward pass (`0.5x(1 + tanh(sqrt(2/π)(x + 0.044715x³)))`)
- **Adam optimizer with weight decay** — bias-corrected momentum, adaptive learning rates
- **Learned position embeddings** (matching actual GPT-1, not sinusoidal)
- **BPE-style vocabulary** construction from character pairs
- **Top-k sampling** for generation
- Per-epoch loss curves and sample generation

---

## Quick Start

**Requirements:** Python 3.8+, NumPy, Matplotlib

```bash
pip install numpy matplotlib
```

**Educational demo (forward pass + attention visualization):**
```bash
python gpt1_from_sutskever30.py
```

**Full training with backpropagation:**
```bash
python gpt1_complete_implementation.py
```

---

## Model Configurations

| Parameter | GPT-1 Original | Demo Version |
|-----------|---------------|--------------|
| vocab_size | 40,000 (BPE) | ~70 (char-level) |
| d_model | 768 | 64–128 |
| num_heads | 12 | 4 |
| num_layers | 12 | 2 |
| d_ff | 3,072 | 256–512 |
| max_seq_len | 512 | 128–256 |
| Parameters | ~117M | ~100K–400K |

---

## Companion Projects

- [sutskever-30-implementations](https://github.com/pageman/sutskever-30-implementations) — Individual NumPy implementations of all 30 papers (100% complete). This is the source repository from which the components here are drawn.
- [Sutskever-Agent](https://github.com/pageman/Sutskever-Agent) — A GitAgent (Claude Sonnet 4.6) that can explain, compare, and generate exercises for any of the 30 implementations. Use it to go deeper on Papers 02, 13, or 27.

---

## Key Equations

**Scaled dot-product attention** (Paper 13):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

**Multi-head attention** (Paper 13):

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

**Language modeling objective** (Paper 27):

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t \mid x_{<t})$$

**GELU activation** (used in GPT-1):

$$\text{GELU}(x) = 0.5x\!\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715x^3)\right)\right)$$

---

## References

1. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised)
2. Vaswani, A., et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
3. Karpathy, A. (2015). [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
4. Gloeckle, F., et al. (2024). [Better & Faster Large Language Models via Multi-Token Prediction](https://arxiv.org/abs/2404.19737)

---

## License

MIT — see [LICENSE](LICENSE)
