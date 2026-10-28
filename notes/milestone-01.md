# Milestone 1: Scaled dot-product attention and masks

## Answers to questions
1. Shape of `Q: (..., T_q, d_k)`, `K: (..., T_k, d_k)` and `V: (..., T_k, d_v)`. <br> 
    - Attention score: Matrix Multiplication of Q and K transpose. So, Attention Score shape: `(..., T_q, T_k)`.
    - Weights: Softmax of scaled Attention score. So, Weights shape: `(..., T_q, T_k)`.
    - Output: Matrix multiplication of Weights and Values. So, Output shape: `(..., T_q, d_v)`

2. In Cross Attention, Queries comes from Decoder, and Key and Values comes from Encoder's output.

3. Softmax computes exp(score) and exp(0)=1 which is an ordinary score. So, the masked key gets a normal share of attention and not None. And, exp(large negative number) is zero, essentially neglecting its impact. We apply masking before softmax because the sum of weights need to be 1. If we apply masking after softmax, masked key stays in the denominator and still controls the other weights.

4. Softmax produces NaN for a row that is entirely `-inf`. Since we fill masked positions with highest negative value, my function produce plain average as softmax of identical values produce identical weights.

5. If the variance of each component of `q` and `k` is 1, their dot product has variance of `d_k`. For `d_k=64`, standard deviation is 8. Dividing by √d_k brings back variance to 1. Large scores push softmax into saturation and saturated softmax has gradients close to zero. It prevents vanishing gradients problem.



## Bugs I hit and their causes

1. The circular `tril` check. Used hard-coded mask.
2. Forgot to set tolerance in `torch.testing.assert_close`. Set `atol` and `rtol`.
3. Added `assert` with `torch.testing.assert_close`. Removed `assert`.
4. Mixed `query` and `key` axes.
5. Filled all values with mask without filtering in causal mask.