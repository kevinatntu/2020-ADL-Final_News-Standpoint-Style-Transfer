# Enhance encoder - using BERT to replace original transformer encoder

Modified from [`Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation` Github repository](https://github.com/Nrgeup/controllable-text-attribute-transfer)

## Command
- Train
  - Full BERT version
    > python main.py --task='news_china_taiwan' --epochs=[NUM_EPOCH] --batch_size=[BATCH_SIZE] --training
  - Fix first 6 layers version
    > python main.py --task='news_china_taiwan' --epochs=[NUM_EPOCH] --batch_size=[BATCH_SIZE] --training
  - Fix last 6 layers version
    > python main.py --task='news_china_taiwan' --epochs=[NUM_EPOCH] --batch_size=[BATCH_SIZE] --training
  - Distill BERT version (get latent from second layer, fix all behind)
    > python main.py --task='news_china_taiwan' --epochs=[NUM_EPOCH] --batch_size=[BATCH_SIZE] --training

- Eval
  - You need to train the model first
  - Taiwan news -> China news
    > python main.py --task='news_china_taiwan' --batch_size=[BATCH_SIZE] --eval_positive
  - China news -> Taiwan news
    > python main.py --task='news_china_taiwan' --batch_size=[BATCH_SIZE] --eval_negative
  


## Change

- Overall: Turn to PyTorch 1.5 version (orginal 0.4)
- model.py: Replace with PyTorch nn.Transformer and huggingface transformer
- data.py: Use Dataloader. only support our own dataset - Taiwan and China news
- main.py: Write correponsding train & eval functions

## Reference

<pre><code>@inproceedings{DBLP:journals/corr/abs-1905-12926,
  author    = {Ke Wang and Hang Hua and Xiaojun Wan},
  title     = {Controllable Unsupervised Text Attribute Transfer via Editing Entangled Latent Representation},
  booktitle = {NeurIPS},
  year      = {2019}
}
</code></pre>




