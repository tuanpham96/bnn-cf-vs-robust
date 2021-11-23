# Experiment 1: Permuted MNIST with natural corruptions

The source notebooks were run on [Kaggle][source]. The inputs were generated for both original MNIST and permuted MNIST following [codes][bnn-meta-paper-repo] of [meta plasticity BNN paper][bnn-meta-paper], then these were saved as well as their [natural corrupted][natcrpt-paper] verions (only 1 level of severity was used for testing for now) adapting the [robustness codes][natcrpt-paper-code], specifically adapting [`corruption.py`][natcrpt-paper-code] to use for monochrome images.

- for variations of `meta` parameter: [Version 2][V2] of [source]:
  - variation of `meta` parameter: `[0.0, 0.7, 1.35]`
  - these following parameters were fixed:

    ``` bash
    --hidden-layers 2048 2048 --lr 0.005 --decay 1e-7 --epochs-per-task 25
    ```

  - the inputs were generated then downloaded in `data/input/pmnist_robustness`
  - the outputs were downloaded into `data/output/exp1-pmnist-robustness`
  - note: the logs for [V2] in the corruption part were not correct due to a typo but the data saved were right. This was fixed for future uses.

[bnn-meta-paper]: https://www.nature.com/articles/s41467-021-22768-y
[bnn-meta-paper-repo]: https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN
[natcrpt-paper]: https://arxiv.org/abs/1903.12261
[natcrpt-paper-repo]: https://github.com/hendrycks/robustness
[natcrpt-paper-code]: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
[source]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust/
[V2]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust/data?scriptVersionId=79007488
