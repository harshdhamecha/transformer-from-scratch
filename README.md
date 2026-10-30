# Transformer from Scratch

[![tests](https://github.com/harshdhamecha/transformer-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/harshdhamecha/transformer-from-scratch/actions/workflows/tests.yml)

The original encoder-decoder Transformer implemented from the paper, without tutorials or reference code.

## Progress

- [x] M1 Scaled dot-product attention and masks
- [x] M2 Multi-head attention
- [x] M3 LayerNorm, feed-forward, positional encoding, embeddings
- [ ] M4 Encoder
- [ ] M5 Decoder
- [ ] M6 Full Transformer
- [ ] M7 Toy task: sequence reversal
- [ ] M8 Translation

## Images
![Positional Encoding Chart](/assets/positional_encoding.png)

## Setup

```bash
conda create -y -n transformer-scratch python=3.15
conda activate transformer-scratch
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install pytest==9.1.1 numpy==2.5.4 matplotlib==3.11.2 ipykernel==7.4.0
pytest
```

## Rules

`nn.MultiheadAttention`, `nn.LayerNorm`, `nn.Transformer*` and
`F.scaled_dot_product_attention` are not used inside the `transformer/` package.
They may be used in `tests/` only as numerical references.
