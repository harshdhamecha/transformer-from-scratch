# Milestone 2: Multi-head attention

## Answers to questions
1. Applying `reshape` directly does not produce the correct values for each tokens. On the other hand, `reshape` to `(B, T, h, d_k)` followed by `transpose` produce the correct result.
2. Sometimes, the tensor's `stride` property changes and that makes the tensor non-contiguous which results into an error. `.contiguous()` makes the tensor contiguous.
3. `W_O` allows information across different heads to interact and blend together. It is learned parameter, so it also let the model know which information from which heads are most important.
4. Projections (Q, K, V, O), Scores (Q * K_t) and (Weights * V) costs the same as we are splitting heads.


## Bugs I hit and their causes
- `super.__init__()` instead of calling `super()`
- `d_model / h` producing a float
- assert `d_model / h` being always true
- packing Q/K/O instead of Q/K/V (a substring filter excluded the wrong layer, and
the shape was still correct)
- the missing `batch_first=True` (PyTorch read T as the batch)
- `=` vs `copy_` (the Parameter type error)
- `out_proj_weight` with an underscore silently creating a new attribute
- the `w_name` attribute lookup
- `key_padding_mask` vs `attn_mask`
- the two disagreeing dropout conditions

## Learning
1. Check whether `d` is divisible by `num_heads`.
2. `Dropout` is used in `attention` block as well, right beffore multiplication with `values`.
3. Copy `weights` from your own module to `torch`'s weights. 
4. Split `d_model` into `heads` and `d_q` and merge them back.
5. MHA `output`'s shape is the same as `input`'s shape. 