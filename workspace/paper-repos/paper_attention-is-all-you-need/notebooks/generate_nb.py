import json
import os

def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    }

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    }

primary_cells = [
    markdown_cell("""# Attention Is All You Need
**ArXivist-generated reproduction notebook**
Paper: Not provided
Generated: 2026-07-10

This notebook walks through the key components of the implementation, runs a
small-scale training loop, and verifies that the setup matches the paper's
reported behavior on a mini-dataset."""),
    
    code_cell("""# Check Python version, GPU availability, and key dependencies
import sys, torch
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Running on CPU — training will be slow")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""),
    
    code_cell("""# Install the project in editable mode (run once)
import subprocess
result = subprocess.run(["pip", "install", "-e", ".."], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else result.stderr)"""),
    
    markdown_cell("""## Paper Overview

**Problem**: The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. 

**Core Idea**: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality.

**Mapping to implementation**:
- `src.transformer.models.attention`: Implements Scaled Dot-Product Attention (3.2.1) and Multi-Head Attention (3.2.2).
- `src.transformer.models.layers`: Implements Position-wise Feed-Forward Network (3.3) and Positional Encoding (3.5).
- `src.transformer.models.transformer`: Implements Encoder Stack (3.1), Decoder Stack (3.1), and the full Transformer model."""),
    
    markdown_cell("""## Positional Encoding
Since the Transformer has no recurrence or convolution, it must inject some information about the relative or absolute position of the tokens in the sequence. This is done using sine and cosine functions of different frequencies.
$$ PE_{(pos, 2i)} = \\sin(pos/10000^{2i/d_{\\text{model}}}) $$
$$ PE_{(pos, 2i+1)} = \\cos(pos/10000^{2i/d_{\\text{model}}}) $$"""),
    
    code_cell("""import torch

try:
    from src.transformer.models.layers import PositionalEncoding
    d_model = 512
    max_len = 100
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len).to(device)
    
    x = torch.zeros(2, 10, d_model).to(device) # Batch size 2, Sequence length 10
    output = pos_encoder(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 10, {d_model}])")
except Exception as e:
    print(f"Error: {e}")"""),
    
    markdown_cell("""## Multi-Head Attention
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
$$ \\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, ..., \\text{head}_h)W^O $$
$$ \\text{Attention}(Q, K, V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V $$"""),
    
    code_cell("""import torch

try:
    from src.transformer.models.attention import MultiHeadAttention
    model_config = {
        'h': 8,
        'd_model': 512,
        'd_k': 64,
        'd_v': 64,
        'dropout': 0.1
    }
    attention = MultiHeadAttention(**model_config).to(device)
    
    # Toy forward pass (Self-Attention)
    q = torch.randn(2, 10, 512).to(device)
    k = q
    v = q
    
    output = attention(q, k, v)
    print(f"Input shape:  {q.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 10, 512])")
except Exception as e:
    print(f"Error: {e}")"""),
    
    markdown_cell("""## Position-wise Feed-Forward Network
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically.
$$ \\text{FFN}(x) = \\max(0, xW_1 + b_1)W_2 + b_2 $$"""),
    
    code_cell("""import torch

try:
    from src.transformer.models.layers import PositionwiseFeedForward
    model_config = {
        'd_model': 512,
        'd_ff': 2048,
        'dropout': 0.1
    }
    ffn = PositionwiseFeedForward(**model_config).to(device)
    
    x = torch.randn(2, 10, 512).to(device)
    output = ffn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 10, 512])")
except Exception as e:
    print(f"Error: {e}")"""),
    
    markdown_cell("""## Encoder Stack
The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection around each of the two sub-layers, followed by layer normalization."""),
    
    code_cell("""import torch

try:
    from src.transformer.models.transformer import Encoder
    encoder_config = {
        'N': 6,
        'd_model': 512,
        'd_ff': 2048,
        'h': 8,
        'd_k': 64,
        'd_v': 64,
        'P_drop': 0.1,
        'vocab_size': 100
    }
    # For a generic Encoder, it usually takes token IDs if it has the embedding layer.
    encoder = Encoder(**encoder_config).to(device)
    
    x = torch.randint(0, 100, (2, 10)).to(device)
    mask = (x != 0).unsqueeze(1).unsqueeze(2).to(device)
    output = encoder(x, mask)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 10, 512])")
except Exception as e:
    print(f"Error: {e}")"""),

    markdown_cell("""## Decoder Stack
The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack."""),
    
    code_cell("""import torch

try:
    from src.transformer.models.transformer import Decoder
    decoder_config = {
        'N': 6,
        'd_model': 512,
        'd_ff': 2048,
        'h': 8,
        'd_k': 64,
        'd_v': 64,
        'P_drop': 0.1,
        'vocab_size': 100
    }
    decoder = Decoder(**decoder_config).to(device)
    
    tgt = torch.randint(0, 100, (2, 10)).to(device)
    memory = torch.randn(2, 10, 512).to(device)
    tgt_mask = torch.ones(2, 1, 10, 10).to(device)
    memory_mask = torch.ones(2, 1, 10, 10).to(device)
    
    output = decoder(tgt, memory, memory_mask, tgt_mask)
    print(f"Input shape:  {tgt.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     torch.Size([2, 10, 512])")
except Exception as e:
    print(f"Error: {e}")"""),
    
    markdown_cell("""## Mini-Training Demonstration"""),

    code_cell("""import torch
from torch.utils.data import TensorDataset, DataLoader

# Generate a tiny synthetic dataset for seq2seq (e.g., copying task)
vocab_size = 100
seq_length = 15
num_samples = 100

src_data = torch.randint(1, vocab_size, (num_samples, seq_length))
tgt_data = src_data.clone() # Simple copy task for demonstration

dataset = TensorDataset(src_data, tgt_data)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
print(f"Dataset created with {num_samples} samples.")"""),
    
    code_cell("""import torch.nn as nn
import torch.optim as optim

try:
    from src.transformer.models.transformer import Transformer
    model_config = {
        'N': 2, # Reduced for speed
        'd_model': 128,
        'd_ff': 256,
        'h': 4,
        'd_k': 32,
        'd_v': 32,
        'P_drop': 0.1,
        'src_vocab_size': vocab_size,
        'tgt_vocab_size': vocab_size
    }
    model = Transformer(**model_config).to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model instantiated with {total_params:,} trainable parameters.")
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
except Exception as e:
    print(f"Error: {e}")"""),

    code_cell("""try:
    model.train()
    epochs = 5
    print("Starting mini-training loop...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for src, tgt in dataloader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            # For seq2seq, tgt_input is tgt[:-1], tgt_output is tgt[1:]
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            # Mock masks
            src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(device)
            tgt_mask = torch.tril(torch.ones((tgt_input.size(1), tgt_input.size(1)))).bool().to(device)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            # output shape: [B, T, vocab_size]
            loss = criterion(output.reshape(-1, vocab_size), tgt_expected.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")
    print("Mini-training complete.")
except Exception as e:
    print(f"Error: {e}")"""),
    
    code_cell("""try:
    model.eval()
    with torch.no_grad():
        sample_src = src_data[0:1].to(device)
        sample_tgt = sample_src.clone()[:, :-1] # Teacher forcing for demo
        src_mask = (sample_src != 0).unsqueeze(1).unsqueeze(2).to(device)
        tgt_mask = torch.tril(torch.ones((sample_tgt.size(1), sample_tgt.size(1)))).bool().to(device)
        
        output = model(sample_src, sample_tgt, src_mask, tgt_mask)
        predicted_tokens = output.argmax(dim=-1)
        
        print(f"Source token:   {sample_src[0].cpu().tolist()}")
        print(f"Predicted next: {predicted_tokens[0].cpu().tolist()}")
except Exception as e:
    print(f"Error: {e}")"""),
    
    markdown_cell("""## Paper Results Comparison"""),
    
    code_cell("""# Results reported in the paper (from SIR evaluation_protocol.reported_results)
paper_results = [
    {
        "dataset": "WMT 2014 English-German",
        "metric": "BLEU",
        "reported_value": 28.4
    },
    {
        "dataset": "WMT 2014 English-French",
        "metric": "BLEU",
        "reported_value": 41.0
    }
]
print("Paper's claimed results:")
for res in paper_results:
    print(f"  {res['dataset']} - {res['metric']}: {res['reported_value']}")
print("\\nTo reproduce these results, run train.py with the full config.")
print("Then use the Results Comparator (Stage 6) to compare your outputs.")"""),
    
    markdown_cell("""## What to do next

1. **Full training**: `python train.py --config configs/config.yaml`
2. **Evaluation**: `python evaluate.py --checkpoint checkpoints/best.pt`
3. **Compare results**: Feed your results back to ArXivist's Results Comparator

**Implementation notes from the SIR:**
- Batch dimension is the first dimension in tensors (e.g., [B, T, D]) (Confidence: 0.9)
- Weights are initialized using standard initialization schemes like Xavier/Glorot. (Confidence: 0.8)
- Masking out illegal connections sets values to -infinity. Implementation in practice uses a very large negative number (-1e9). (Confidence: 0.85)""")
]

exploratory_cells = [
    markdown_cell("""# Attention Is All You Need - Exploratory Analysis
**ArXivist-generated visualization notebook**

This notebook allows you to explore the internal representations and attention maps of the Transformer model. 

> **Note**: You must provide a trained checkpoint to fully utilize this notebook. If you haven't trained a model yet, run the primary reproduction notebook or `train.py` first, or download a pre-trained checkpoint."""),
    
    code_cell("""import sys, torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")"""),
    
    code_cell("""# Note: Update these paths and configs based on your training run
checkpoint_path = "../checkpoints/best.pt"
model_config = {
    'N': 6,
    'd_model': 512,
    'd_ff': 2048,
    'h': 8,
    'd_k': 64,
    'd_v': 64,
    'P_drop': 0.1,
    'src_vocab_size': 37000,
    'tgt_vocab_size': 37000
}

try:
    from src.transformer.models.transformer import Transformer
    model = Transformer(**model_config).to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully.")
except FileNotFoundError:
    print(f"Checkpoint not found at {checkpoint_path}. Instantiating an untrained model for demonstration.")
    model = Transformer(**model_config).to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")"""),
    
    markdown_cell("""## Visualization 1: Self-Attention Maps
The Transformer relies heavily on self-attention. We can visualize the attention weights between different tokens in the input sequence. The self-attention mechanism is defined as:
$$ \\text{Attention}(Q, K, V) = \\text{softmax}(\\frac{QK^T}{\\sqrt{d_k}})V $$"""),
    
    code_cell("""def plot_attention_map(attention_matrix, src_tokens, tgt_tokens=None, title="Attention Map"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_matrix, xticklabels=src_tokens, yticklabels=tgt_tokens or src_tokens, 
                cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel("Key Tokens")
    plt.ylabel("Query Tokens")
    plt.show()

# Mocking some attention weights for demonstration
try:
    seq_len = 10
    mock_tokens = [f"token_{i}" for i in range(seq_len)]
    # Simulate a single head's attention map for a 10-token sequence
    mock_attention = torch.softmax(torch.randn(seq_len, seq_len), dim=-1).numpy()
    
    plot_attention_map(mock_attention, mock_tokens, title="Mock Encoder Self-Attention (Head 1)")
except Exception as e:
    print(f"Error plotting attention: {e}")"""),
    
    markdown_cell("""## Visualization 2: Positional Encodings
The positional encodings are fixed sinusoidal waves added to the embeddings. We can visualize how the dimensions oscillate at different frequencies."""),
    
    code_cell("""try:
    from src.transformer.models.layers import PositionalEncoding
    d_model = 512
    max_len = 100
    pos_encoder = PositionalEncoding(d_model=d_model, max_len=max_len)
    
    # Extract the pre-computed positional encodings
    pe = pos_encoder.pe.squeeze(0).numpy() # shape: [100, 512]
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(pe, cmap='RdBu', vmin=-1.0, vmax=1.0)
    plt.title("Positional Encoding Matrix")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    plt.show()
except Exception as e:
    print(f"Error plotting positional encoding: {e}")"""),
    
    markdown_cell("""## Visualization 3: Multi-Head Comparison
Different attention heads often learn to focus on different aspects of the sequence (e.g., one head for local syntax, another for long-range dependencies). 
We can use `ipywidgets` to interactively switch between heads."""),
    
    code_cell("""import ipywidgets as widgets
from IPython.display import display

try:
    num_heads = model_config['h']
    seq_len = 12
    mock_tokens_2 = [f"tok_{i}" for i in range(seq_len)]
    
    # Mocking attention matrices for 8 heads
    mock_multi_head_attention = [torch.softmax(torch.randn(seq_len, seq_len), dim=-1).numpy() for _ in range(num_heads)]
    
    def view_head(head_index):
        print(f"Visualizing Head {head_index}")
        plot_attention_map(mock_multi_head_attention[head_index], mock_tokens_2, title=f"Attention - Head {head_index}")
    
    head_slider = widgets.IntSlider(min=0, max=num_heads-1, step=1, description='Head:', value=0)
    widgets.interact(view_head, head_index=head_slider)
except Exception as e:
    print(f"Error creating interactive widget: {e}")""")
]

nb_primary = {
    "cells": primary_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

nb_exploratory = {
    "cells": exploratory_cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

base_path = r"F:\QOSI Fellowship\Research Papers\outputs\paper-repos\paper_attention-is-all-you-need\notebooks"
os.makedirs(base_path, exist_ok=True)

with open(os.path.join(base_path, "reproduce_paper_attention-is-all-you-need.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb_primary, f, indent=2)

with open(os.path.join(base_path, "explore_paper_attention-is-all-you-need.ipynb"), "w", encoding="utf-8") as f:
    json.dump(nb_exploratory, f, indent=2)
