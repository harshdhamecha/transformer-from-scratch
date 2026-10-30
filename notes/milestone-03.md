# Milestone 3: LayerNorm, feed-forward, positional encoding, embeddings

## Answers to questions
1. `BatchNorm`'s statistics get contaminated by padding, while `LayerNorm` normalizes each token using only its own features. A real token's `LayerNorm` output depends only on that token's own `d_model` values, so changing the padding or the other sentences in the batch can't affect it. At inference, you may have a batch of one, where batch statistics don't work, so `BatchNorm` has to rely on running averages from training. `LayerNorm` behaves the same in training and inference.
2. PyTorch's `nn.Parameter` is trained by Optimizer. But, we don't want to train Positional Encoding vectors. But we do want it to move it to the device and in`state_dict` and that's why we register buffer.
3. To make the token embeddings as large as the positional encodings, so that adding the two doesn't bury the word's meaning under its position.
4. Positions 3 and 5 relate exactly the way positions 103 and 105 do. It is bounded and has no parameters.

## Bugs I hit and their causes
- Shape mismatch in `LayerNorm` without `keepdim=True`.
- Wrong variance estimator without `correction=0`. Check `biased` vs `unbiased`. We have to use `biased`.
- `math.sqrt` for scalar and `torch.sqrt` for tensors.
- `2 * values` with the current `values` array produced wrong results.
- `torch.where` did not work and thus used slicing.
