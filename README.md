Begin: I still don't have an intuitive understanding of attention. I have watched countless videos but I still can't explain it without fumbling at some point and being confused.  

Idea: Implement common attention mechanisms to understand.  

End: I understand attention from a very intuitive level. I had to brush up on matrix multiplication, vectorization, etc to implement this at a relatively lower level. I'd highly suggest anyone else to try this exercise. If your intention is to learn how attention works, please don't look at my code unless you've spent a couple hours trying to implement it yourself. If you just want a layer that can contextualize your embeddings use the `SelfAttention` module from `SelfAttention.py`, or if you want trainable parameters in your attention block, use `KVQ_selfattention` from `KVQ_selfattention.py`.  

You can also look at `self_attention_forloop.py` to see an unefficient (but easier to read and comprehend) implementation of self-attention.