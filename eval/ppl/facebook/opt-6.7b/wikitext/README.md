---
license: other
tags:
- generated_from_trainer
datasets:
- wikitext
model-index:
- name: wikitext
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wikitext

This model is a fine-tuned version of [facebook/opt-6.7b](https://huggingface.co/facebook/opt-6.7b) on the wikitext wikitext-2-v1 dataset.
It achieves the following results on the evaluation set:
- eval_loss: 7.3174
- eval_accuracy: 0.2004
- eval_runtime: 196.5549
- eval_samples_per_second: 1.267
- eval_steps_per_second: 0.321
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 4
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1.0

### Framework versions

- Transformers 4.24.0
- Pytorch 2.0.0+cu117
- Datasets 2.6.1
- Tokenizers 0.11.0
