# Notes

All the output data are (or will be) at the [google-drive][data-drive] as tar files:

- These files are assumed to be in `data`.
- If they start with `input` then they should be in `data/input`
- If they start with `output` then they should be in `data/output`

The files in this folder describe the details about the experiments.

- `exp0-paper-rerun`
  - Rerun the source [paper] original [codes][paper-repo] to see whether results are replicable.
  - Most updated file in [data-drive]: `output-exp0-pmnist-paper-rerun-20211108.tar.xz`
- `exp1-pmnist-robustness`
  - First test to run permuted MNIST with [natural corruptions][natcrpt-paper] and FGSM attacks.
  - Most updated file in [data-drive]: `output-exp1-pmnist-robustness-20211108.tar.xz`
- `exp2-pmnist-dq`
  - Integrate metaplasticity training with [Lipschitz regularization][dq-paper] for pMNIST task, tested for both natural corruptions, as well as a few more adversarial attacks from [`foolbox`][foolbox].
  - Most updated file in [data-drive]: `output-exp2-pmnist-robust-dq-20211126.tar.xz`
  - This is more thorough than `exp1-pmnist-robustness`

[paper-repo]: https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN
[paper]: https://www.nature.com/articles/s41467-021-22768-y
[data-drive]: https://drive.google.com/file/d/1CxCEhm1izTdB1pp4A_J8Vip3XIRNyB6z/view?usp=sharing
[foolbox]: https://github.com/bethgelab/foolbox
[natcrpt-paper]: https://arxiv.org/abs/1903.12261
[dq-paper]: https://arxiv.org/pdf/1904.08444.pdf
