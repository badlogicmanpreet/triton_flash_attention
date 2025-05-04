# Understanding Transformer Attention: A Beginner's Guide

This tutorial breaks down the transformer architecture, focusing on multi-head attention mechanisms as implemented in models like GPT-2. We'll use concrete examples and intuitive explanations to make these complex concepts accessible.

## Table of Contents
1. [Transformer Architecture Overview](#transformer-architecture-overview)
2. [Understanding Transformer Blocks](#understanding-transformer-blocks)
3. [Self-Attention Mechanism](#self-attention-mechanism)
4. [Multi-Head Attention](#multi-head-attention)
5. [Practical Example: Self-Attention Calculation](#practical-example-self-attention-calculation)
6. [Summary](#summary)

## Transformer Architecture Overview

A modern transformer model like GPT-2 consists of stacked layers (blocks), each containing attention mechanisms and feed-forward neural networks. Here's a simplified view:

```
Input Embeddings + Positional Encodings
↓
Block 1 → Self-Attention → Feed-Forward Network
↓
Block 2 → Self-Attention → Feed-Forward Network
↓
...
↓
Block N → Self-Attention → Feed-Forward Network
↓
Output Layer
```

The two key components of each block are:
- **Self-Attention**: Allows tokens to "look at" other tokens in the sequence
- **Feed-Forward Network**: Processes each token independently

## Understanding Transformer Blocks

Transformer blocks are processed **sequentially**, not in parallel. The output of one block becomes the input to the next.

In the case of GPT-2 (base model), there are 12 blocks stacked on top of each other, creating a deep network:

```python
# From the code
n_layer: int = 12  # number of transformer blocks
```

Each block enriches the representation of the input sequence, with higher blocks capturing increasingly complex patterns and relationships between tokens.

## Self-Attention Mechanism

Self-attention is the heart of transformers. It allows each token in a sequence to "pay attention" to all other tokens (in GPT, only to previous tokens due to the causal mask).

### The Key Components

For each token, we calculate three vectors:
- **Query (Q)**: What the token is "looking for"
- **Key (K)**: What the token "offers" to be found
- **Value (V)**: The actual information the token passes when attended to

These vectors are created through learned linear transformations of the input embeddings:

```python
# From the code
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
qkv = self.c_attn(x)
q, k, v = qkv.split(self.n_embd, dim=2)
```

### Attention Calculation

The attention mechanism calculates how much each token should attend to every other token:

1. Compute attention scores: **S = Q × K^T** (matrix multiplication)
2. Scale the scores: **S = S / √d** (where d is the dimension of the key vectors)
3. Apply masking (for causal attention in GPT models)
4. Apply softmax to get attention weights
5. Compute the weighted sum of values: **Output = Attention_Weights × V**

```python
# From the code
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
y = att @ v
```

## Multi-Head Attention

Instead of performing a single attention function, transformers use **multiple attention heads** in parallel. In GPT-2 base, there are 12 attention heads:

```python
# From the code
n_head: int = 12  # number of attention heads
```

### Why Multiple Heads?

Multi-head attention allows the model to:
1. **Focus on different aspects** of relationships between tokens
2. **Attend to different positions** simultaneously
3. **Increase representation capacity** without increasing depth
4. **Create specialized attention patterns** for different types of relationships

### How Multi-Head Attention Works

1. The input is projected into different subspaces for each head
2. Each head performs its own self-attention calculation
3. The outputs from all heads are concatenated and projected back to the original dimension

```python
# From the code
# Split into heads
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

# Compute attention for each head
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# ... mask and softmax ...
y = att @ v

# Combine heads
y = y.transpose(1, 2).contiguous().view(B, T, C)
y = self.c_proj(y)  # Final projection
```

In GPT-2 base, the embedding dimension is 768, and with 12 heads, each head works with 64-dimensional vectors (768/12 = 64).

## Practical Example: Self-Attention Calculation

Let's walk through a concrete example of how self-attention is calculated for a single head:

Imagine we have a sequence of 4 tokens, each represented by a 64-dimensional vector:

1. We have matrices:
   - Q (4×64): Each row is a query vector for one token
   - K (4×64): Each row is a key vector for one token
   - V (4×64): Each row is a value vector for one token

2. Calculate attention scores S = Q × K^T (resulting in a 4×4 matrix)
   - For each token i, take its query vector qi (row i of Q)
   - Multiply it with each column of K^T (each row of K)
   - This gives a row in S with 4 scores, showing how much token i attends to each token

   For example, to calculate how much token 1 attends to token 2:
   ```
   S[1,2] = Q[1,:] · K[2,:]  (dot product of row 1 of Q and row 2 of K)
   ```

3. Scale the scores: S = S / √64 = S / 8

4. Apply causal masking (in GPT models): Set S[i,j] = -∞ for all j > i
   This creates a lower triangular matrix, ensuring tokens only attend to themselves and previous tokens.

5. Apply softmax to each row of S to get attention weights
   - Each row sums to 1, representing a probability distribution
   - The first row has only one non-zero value (attending only to itself)
   - The last row has values for attending to all tokens

6. Calculate the weighted sum of values:
   - For each token i, take its row of attention weights
   - Compute the weighted sum of all value vectors using these weights
   - This gives the output for token i

From the log file, we can see an example of attention weights after softmax:
```
[[1.0000, 0.0000, 0.0000, 0.0000],
 [0.6405, 0.3595, 0.0000, 0.0000],
 [0.3258, 0.3123, 0.3619, 0.0000],
 [0.0247, 0.0236, 0.0253, 0.9264]]
```

This shows:
- Token 1 fully attends to itself (can't see other tokens)
- Token 2 attends 64% to token 1, 36% to itself
- Token 3 attends roughly equally to tokens 1, 2, and itself
- Token 4 attends most strongly to itself, but also somewhat to tokens 1, 2, and 3

## Summary

1. **Transformer Architecture**: Consists of stacked blocks processed sequentially
2. **Self-Attention**: Allows tokens to gather information from other tokens
3. **Multi-Head Attention**: Enables the model to focus on different aspects of relationships simultaneously
4. **Query, Key, Value Vectors**: Different projections of the input used to calculate attention
5. **Attention Calculation**: Matrix multiplication of queries and keys, followed by scaling, masking, and softmax
6. **Block Processing**: Each block enriches the representation before passing it to the next block

The power of transformers comes from this ability to model complex relationships between tokens, with each layer and each attention head capturing different aspects of these relationships.
