# Milestone 3: LayerNorm, feed-forward, positional encoding, embeddings

## Concepts

## Answers to questions

## Bugs I hit and their causes
- Shape mismatch in `LayerNorm` without `keepdim=True`.
- Wrong variance estimator without `correction=0`. Check `biased` vs `unbiased`. We have to use `biased`.
- `math.sqrt` for scalar and `torch.sqrt` for tensors.

## What I learned
