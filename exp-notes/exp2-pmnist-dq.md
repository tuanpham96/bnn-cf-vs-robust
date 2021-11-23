# Experiment 2: Permuted MNIST with natural corruptions, adversarial attacks with both metaplasticity training and Lipschitz regularization (defensive quantization DQ)

The source notebooks were run on [Kaggle][source]. The inputs were generated for both original MNIST and permuted MNIST following [codes][bnn-meta-paper-repo] of [meta plasticity BNN paper][bnn-meta-paper], then these were saved as well as their [natural corrupted][natcrpt-paper] verions (only 1 level of severity was used for testing for now) adapting the [robustness codes][natcrpt-paper-code], specifically adapting [`corruption.py`][natcrpt-paper-code] to use for monochrome images. To avoid having to re-generate the input data (actually could take 1-1.5 hrs), the input data are from [version 2][V2] in [exp-1](exp1-pmnist-robustness.md).

In this experiment, in addition to meta-plasticity training, the Lipschitz regularization was also included from the defensive quantization (DQ) [paper][dq-paper] (initially implemented to increased quantized models' robustness to adversarial attacks), in which the parameter `beta_dq` controls the scaling factor for the regularization. Plus, aside from testing for natural corruptions, adversarial attacks from the toolbox [foolbox] were also integrated and tested after the model was trained with meta-plasticity and DQ.

- The experiment was run in version 3-6 (due to Kaggle limit per instance) ([V3], [V4], [V5], [V6]) of [source]:
  - `meta` variations: `[0.0, 0.7, 1.35]`
  - `beta_dq` variations: `[0, 3e-4, 6e-4]` (based on choices in [dq-paper])
  - fixed hyperparameters:

    ``` bash
    --hidden-layers 1024 1024
    --lr 0.005 --decay 1e-7
    --epochs-per-task 40
    --batch-size 100
    ```

  - the inputs were generated from [version 2][V2] in [`exp1`](exp1-pmnist-robustness.md).
    - additionally the settings for adversarial attacks for [`foolbox`][foolbox] functions (integrated into `metadata.yaml` of the input) was the following (taken from the  `add_advattack_meta.py` script inside the folder)

    ``` python
    task_metadata['adversarial_attacks'] = dict(
        attacks = [
            'FGSM',
            'LinfBasicIterativeAttack',
            'LinfAdditiveUniformNoiseAttack',
            {'LinfDeepFoolAttack': dict(steps = 40)},
            {'DDNAttack': dict(steps = 20)},
            {'LinfPGD':  dict(steps = 20)}
        ],
        epsilons = [0.0] \
               + (np.array([[1,2,4,8]]) * np.array([1e-4, 1e-3])[:, None]).flatten().tolist() \
               + (np.logspace(0,3.5,15,base=2.0)*1e-2).round(3).tolist() \
               + [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    )
    ```

    - to read more about these `foolbox` functions' description, please do to their [attacks API][foolbox-attacks]
  - the outputs were downloaded into `data/output/exp2-pmnist-robust-dq`

[bnn-meta-paper]: https://www.nature.com/articles/s41467-021-22768-y
[bnn-meta-paper-repo]: https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN
[natcrpt-paper]: https://arxiv.org/abs/1903.12261
[natcrpt-paper-repo]: https://github.com/hendrycks/robustness
[natcrpt-paper-code]: https://github.com/hendrycks/robustness/blob/master/ImageNet-C/imagenet_c/imagenet_c/corruptions.py
[foolbox]: https://github.com/bethgelab/foolbox
[foolbox-attacks]: https://foolbox.readthedocs.io/en/stable/modules/attacks.html#
[dq-paper]: https://arxiv.org/pdf/1904.08444.pdf
[source]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust/
[V2]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust/data?scriptVersionId=79007488
[V3]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust?scriptVersionId=80532183
[V4]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust?scriptVersionId=80539422
[V5]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust?scriptVersionId=80548447
[V6]: https://www.kaggle.com/penguinsfly/bnn-cf-vs-robust?scriptVersionId=80603416