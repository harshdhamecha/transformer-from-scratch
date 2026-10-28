# Transformer from Scratch: Milestones

The original encoder-decoder Transformer ("Attention Is All You Need"), implemented from the paper without tutorials or reference code.

Every milestone has the same structure:

- **Goal**: what this milestone achieves.
- **Build**: the components to write.
- **Know**: questions to answer in `notes/milestone-XX.md` before moving on.
- **Done when**: tests that must pass. Do not start the next milestone until all of them pass.

## Ground rules

- Inside `transformer/`: `nn.Linear`, `nn.Embedding`, `nn.Dropout`, and basic tensor operations are allowed. `nn.MultiheadAttention`, `nn.LayerNorm`, `nn.Transformer*` and `F.scaled_dot_product_attention` are **not** allowed.
- Inside `tests/`: those modules may be used only as numerical references to check your outputs.
- Mask convention for the whole project: **boolean, `True` = allowed to attend, `False` = blocked.**

## Notation

| Symbol | Meaning |
|---|---|
| B | Batch size |
| T | Sequence length (T_q = number of queries, T_k = number of keys/values) |
| d_model | Embedding size (512 in the paper) |
| h | Number of attention heads (8 in the paper) |
| d_k | Length of each query/key vector (d_model / h = 64) |
| d_v | Length of each value vector (equal to d_k here) |
| d_ff | Hidden size of the feed-forward layer (2048) |
| N | Number of encoder/decoder layers (6) |

---

## M1: Scaled dot-product attention and causal mask

**Goal:** A correct, well-tested attention function, plus the causal mask needed to test it.

**Build:**

In `transformer/attention.py`:

- `attention(q, k, v, mask=None)`, returning `(output, weights)`.
  - Input shapes: `q` is `(..., T_q, d_k)`, `k` is `(..., T_k, d_k)`, `v` is `(..., T_k, d_v)`. The `...` means any number of leading dimensions (batch, later also heads).
  - Steps: scores = Q·Kᵀ → divide by √d_k → apply mask → softmax over the last dimension → weights·V.
  - Every operation must act on the **last two dimensions only**.
  - Output: `(..., T_q, d_v)`. Weights: `(..., T_q, T_k)`.
  - Masked positions get the most negative value the dtype supports (not `-inf`, not `-1e9`), so a fully masked row cannot produce NaN.

In `transformer/masks.py`:

- `causal_mask(T)`: boolean `(T, T)` tensor, `True` on and below the diagonal.

(The padding mask is deferred to M4, where token IDs and padded batches first appear.)

**Know:**

1. For T_q = 5, T_k = 7, d_k = d_v = 4 and batch B: what are the shapes of the scores, the weights and the output?
2. Where in the Transformer do queries come from one sequence and keys/values from another?
3. Why does masking set scores to a large negative value instead of 0? Why is it done before softmax rather than after?
4. What does softmax produce for a row that is entirely `-inf`? What does *your* function produce for a fully masked row, and why?
5. Why divide by √d_k? Give the variance argument and name the specific gradient problem it prevents.

**Done when** (all in `tests/test_attention.py` and `tests/test_masks.py`):

1. **Shapes:** with q `(2, 5, 4)`, k `(2, 7, 4)`, v `(2, 7, 6)`, output is `(2, 5, 6)` and weights are `(2, 5, 7)`. (Using d_v ≠ d_k here catches code that silently assumes they are equal.)
2. **Rows sum to 1:** every row of the weights sums to 1 within `1e-6`.
3. **Causal zeros:** with T = 6 and `causal_mask(6)`, every weight above the diagonal is exactly 0.
4. **No future leakage:** with a causal mask, change K and V at position 5 only. Outputs at positions 0–4 are unchanged.
5. **Masked keys get zero weight:** with a hand-written mask that blocks the last 2 keys, every query's weights on those 2 keys are exactly 0.
6. **Fully masked row:** a row with every position masked produces no NaN in the output.
7. **Reference match:** output matches `F.scaled_dot_product_attention` within `1e-5`, both without a mask and with a causal mask. (Check what `True` means in its mask argument before comparing.)
8. **Extra dimensions:** q, k, v of shape `(2, 3, 5, 4)` (3 stands in for heads) work without any code changes.
9. **Causal mask helper:** `causal_mask(4)` equals the expected 4×4 lower-triangular pattern, with dtype `torch.bool`.

---

## M2: Multi-head attention

**Goal:** Many attention heads running in parallel, each free to learn a different kind of relationship.

**Build:** `MultiHeadAttention(d_model, h, dropout)` in `attention.py`:

- Four linear layers: W_Q, W_K, W_V, W_O.
- Split heads: `(B, T, d_model)` → `(B, h, T, d_k)`.
- Call your M1 `attention` function.
- Merge heads: back to `(B, T, d_model)`, then apply W_O.
- Separate query, key and value inputs, so the same module serves self-attention and cross-attention.
- Accept masks of shape `(B, 1, T_k)`, `(T_q, T_k)` or `(B, T_q, T_k)` and insert the heads dimension so they broadcast to `(B, h, T_q, T_k)`.

**Know:**

1. Why reshape and then transpose, rather than reshape straight into `(B, h, T, d_k)`?
2. Why is `.contiguous()` sometimes needed before merging heads?
3. What does W_O do that concatenating the heads alone doesn't?
4. Why does splitting d_model across h heads cost about the same compute as one big head?

**Done when:**

1. Output shape equals the query input shape, including when T_q ≠ T_k.
2. Parameter count is exactly `4·d_model² + 4·d_model` (compute it by hand first).
3. With weights copied into `nn.MultiheadAttention(batch_first=True)`, outputs match within `1e-5`. Note that it packs W_Q, W_K, W_V into one `in_proj_weight`, and its `key_padding_mask` uses the opposite convention (`True` = ignore).
4. With h = 1, the result equals a single-head computation done by hand with the same weights.
5. The no-future-leakage test from M1 still passes through the full module.

---

## M3: Supporting layers

**Goal:** Every remaining building block, each tested in isolation.

**Build** (in `layers.py`):

- `LayerNorm(d_model, eps)`: learnable gamma (init 1) and beta (init 0), normalizing over the last dimension.
- `FeedForward(d_model, d_ff, dropout)`: d_model → d_ff → d_model with ReLU and dropout.
- `PositionalEncoding(d_model, max_len, dropout)`: sinusoidal, stored with `register_buffer`, added to the input, followed by dropout.
- `Embeddings(vocab_size, d_model)`: embedding lookup multiplied by √d_model.

**Know:**

1. LayerNorm normalizes over features; BatchNorm normalizes over the batch. Why is LayerNorm the better fit for variable-length sequences?
2. Why is the positional encoding a buffer and not a parameter?
3. Why multiply the embeddings by √d_model?
4. Why sinusoids? What property lets the model reason about relative positions?

**Done when:**

1. LayerNorm matches `F.layer_norm` within `1e-5`.
2. Positional encoding at position 0 is `[0, 1, 0, 1, …]`.
3. The positional encoding has no gradient and does not appear in `model.parameters()`.
4. All output shapes are correct.
5. A plot of the positional encoding matrix is saved for the README.

---

## M4: Encoder

**Goal:** The encoder half of the model.

**Build:**

- In `masks.py`: `padding_mask(tokens, pad_id)`. Takes token IDs of shape `(B, T)` (integers from the vocabulary, before embedding) and the ID the vocabulary assigns to `<pad>`. Returns a boolean `(B, 1, T)` tensor, `True` where the token is not padding, on the same device as `tokens`.
- In `encoder.py`:
  - `EncoderLayer`: self-attention, then feed-forward. Each is wrapped as `x = LayerNorm(x + Dropout(sublayer(x)))` (the paper's post-norm arrangement).
  - `Encoder`: embeddings + positional encoding, then N stacked layers.

**Know:**

1. What problem do residual connections solve?
2. Post-norm vs pre-norm: what is the difference, and why did later models move to pre-norm?
3. Why does a batch of sentences need padding at all?
4. Why is the padding mask `(B, 1, T)` and not `(B, T, T)`? Which dimension does broadcasting stretch?

**Done when:**

1. `padding_mask` returns the right values, shape `(B, 1, T)` and dtype for a hand-written batch.
2. Shapes are preserved through the stack.
3. In eval mode, two identical forward passes give identical outputs (dropout is off).
4. **Padding leak test:** change only the pad token IDs in a sentence. Outputs at real-token positions are unchanged.

---

## M5: Decoder

**Goal:** The decoder half, the most intricate part of the model.

**Build** (in `decoder.py`):

- `DecoderLayer` with three sublayers:
  1. Masked self-attention: target padding mask AND causal mask.
  2. Cross-attention: queries from the decoder, keys and values from the encoder output ("memory"), masked by the source padding mask.
  3. Feed-forward.
- `Decoder` stack.
- Add a helper in `masks.py` that combines the target padding mask and the causal mask into one `(B, T, T)` mask.

**Know:**

1. Why does decoder self-attention need both masks, while cross-attention needs only the source padding mask?
2. What exactly is "memory", and why is it computed only once per source sentence?

**Done when:**

1. **Causality through the whole stack:** changing target token 5 does not change decoder outputs at positions 0–4.
2. Changing the memory does change the outputs (cross-attention is wired in).
3. Changing source pad tokens does not change the outputs.

---

## M6: Full Transformer

**Goal:** One model that takes source and target tokens and returns logits.

**Build** (in `model.py`):

- `Transformer` combining source and target embeddings, encoder, decoder, and a final linear layer to vocabulary size.
- Weight tying between the target embedding and the output layer.
- Mask construction inside `forward`.
- Xavier initialization.
- Separate `encode()` and `decode()` methods (needed for generation).

**Know:**

1. What is weight tying, and why does it help?
2. Why is there no softmax inside the model?

**Done when:**

1. Output shape is `(B, T_tgt, vocab_size)`.
2. Parameter count with the paper's base config is in the paper's ballpark (it depends on vocab size).
3. **Overfit one batch:** training repeatedly on one batch of ~8 sentence pairs drives the loss close to 0. A model that can't memorize 8 sentences is broken.

---

## M7: Toy task, sequence reversal

**Goal:** Prove the full training pipeline works on a task with a known correct answer.

**Build** (in `experiments/reverse_task/`):

- Synthetic data: random digit sequences with BOS, EOS and PAD tokens; the target is the reversed sequence.
- Teacher forcing: decoder input = target shifted right by one; labels = target.
- Cross-entropy with `ignore_index=PAD`.
- Adam with the paper's warmup learning-rate schedule.
- Greedy decoding.
- Small config: d_model 128, 2 layers, 4 heads.

**Know:**

1. Why is the decoder input shifted right?
2. What happens to training if you forget `ignore_index`?
3. Why does the warmup schedule exist?

**Done when:** More than 99% exact-match accuracy on held-out sequences within minutes of training. A failure here means a bug in the model or training loop, not in the data.

---

## M8: Translation

**Goal:** Real results on real data.

**Build** (in `experiments/translation/`):

- Dataset: Multi30k German–English, or an English–Hindi corpus (check availability first).
- BPE tokenizer trained with the Hugging Face `tokenizers` library.
- Batching sentences of similar length together to reduce padding.
- Training loop with validation loss, checkpointing and label smoothing (0.1).
- Greedy decoding, BLEU via `sacrebleu`, and cross-attention heatmaps.

**Know:**

1. What does label smoothing do, and why does it hurt perplexity but help BLEU?
2. When and why does greedy decoding fail?

**Done when:**

1. Translations of unseen sentences are mostly sensible.
2. BLEU is reported honestly.
3. At least one cross-attention heatmap is in the README.
4. Final weights are pushed to the Hugging Face Hub and linked from the README.

---

## Optional, after M8

- Pre-norm flag, with a training comparison against post-norm.
- Beam search.
- KV cache for faster decoding.

---

## Workflow for every milestone

1. Create a branch: `git checkout -b m1-attention`.
2. Commit the tests first (`test: ...`), then the implementation that makes them pass (`feat: ...`).
3. Commit each bug fix separately, with the cause in the message (`fix: transpose last two dims of K, not dims 1 and 2`).
4. Fill in `notes/milestone-XX.md` and commit it (`docs: M1 notes`).
5. Open a PR into `main` with: **What** was built, **Tests** and what each proves, **Bugs** hit and why, **Open questions**.
6. Merge with a merge commit (not squash), tag it (`git tag -a m1 -m "M1: scaled dot-product attention"`), tick the milestone in the README, delete the branch.

| Milestone | Branch | Tag | Add to README |
|---|---|---|---|
| M1 | `m1-attention` | `m1` | CI badge |
| M2 | `m2-multihead` | `m2` | — |
| M3 | `m3-layers` | `m3` | Positional encoding plot |
| M4 | `m4-encoder` | `m4` | — |
| M5 | `m5-decoder` | `m5` | — |
| M6 | `m6-model` | `m6` | Parameter count; overfit-one-batch loss curve |
| M7 | `m7-reverse` | `m7` | Accuracy and loss curve |
| M8 | `m8-translation` | `m8` | BLEU, sample translations, heatmaps, Hub link |