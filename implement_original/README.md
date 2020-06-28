# Controllable Text Attribute Transfer

This is our implementation of the paper ["Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation"](https://arxiv.org/abs/1905.12926). The main auto-encoder is inherited from the class `torch.nn.Transformer` . 

## How to run

### Dependencies

```
torch
transformers
nltk
tqdm
```

### Training

* Try Yelp dataset: 

```
python main.py --task yelp
```

* Run Task1 (國臺辦 v.s. 臺獨聯盟): 

```
python main.py --task task1
```

* Run Task2 (國臺辦 v.s. 臺灣新聞): 

```
python main.py --task task2
```
* Run Task3 (國臺辦 v.s. 臺獨聯盟, 臺灣新聞=0.5): 

```
python main.py --task task3
```


## reference

* https://github.com/Nrgeup/controllable-text-attribute-transfer
* https://pytorch.org/tutorials/beginner/transformer_tutorial.html
* https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer