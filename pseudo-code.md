```python
"""
Input:
    q (tensor): the recomputed q
    k (tensor): the recomputed k
    v (tensor): the recomputed v
    kv_cache (`tuple` of tensor): the stored KV cache (already concatenated with dummy cache)
    attn_metadata (`dict`): other parameters, including selected token ids.
Output:
    o (tensor): the output of attention algorithm
"""
def attn_fwd_with_pic(q, k, v, kv_cache, attn_metadata):
    key_cache, value_cache = kv_cache
    # get the indices of important tokens, which are computed after tokenization
    imp_indices = attn_metadata["imp_indicies"]
    # replace the old KV cache with recomputed one
    key_cache[imp_indices] = k
    value_cache[imp_indices] = v
    k = key_cache
    v = value_cache

    # attention algorithm
    o = attn_fwd_func(q, k, v, attn_metadata)
    return o
```
