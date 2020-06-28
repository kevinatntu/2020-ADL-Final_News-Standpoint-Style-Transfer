# Controllable Text Attribute Transfer

This is our implementation of the paper ["Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation"] using GRU-based autoencoder instead of transformer.

## How to run

### Dependencies

```
torch
jeiba
nltk
tqdm
```

### Training

follow the ipynb in this folder step by step to generate weight matrix, word2id, id2word, and train the model. You can do the label changing job when autoencoder and classifier are well trained.

