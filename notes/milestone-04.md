# Milestone 4: Encoder

## Answers to questions
1. Residual connections solves vanishing gradients problem. As we passes through deep layers, the gradients becomes tiny and doesn't contribute to any learning. 
2. The tensor must be rectangular and to match the length of shorter sentence to the longest sentence, we apply padding. 

## Bugs I hit and their causes
- Only one `EncoderLayer` instance
- `ModuleList` built in `forward` from layer outputs, resulted into `TypeError:...`.
- `vocab_size` used as the positional encoding `max_len`.
- `max_len` smaller than `T`.
- Parameter formula missing embeddings.

## What I learned
