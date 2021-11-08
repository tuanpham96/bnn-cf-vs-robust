# Experiment 0: Rerun from paper source code

The source notebooks were run on [Kaggle][source] by basically running some of the [codes][paper-repo] on the original [paper] on Synaptic metaplasticity in BNN (with some modifications).

Then I downloaded data to `data/output/exp0-pmnist-paper-rerun`

- `data/output/exp0-pmnist-paper-rerun/vary-meta`: [Version 1][V1] of [source]
  - variation of `meta` parameter: `[0.0, 0.5, 1.0, 1.35, 2]`
  - these following parameters were fixed:

    ``` bash
    --hidden-layers 1024 1024 --lr 0.005 --decay 1e-7 --epochs-per-task 25
    --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
    ```

  - data downloaded from `SynapticMetaplasticityBNN/Continual_Learning_Fig-2abcdefgh-3abcd-5cde/results` of [V1] (except for `16-01-01_gpu0`)
- `data/output/exp0-pmnist-paper-rerun/vary-hidden`: [Version 2][V2] of [source]
  - variation of size of hidden layers: `[512, 1024, 2048, 4096]`
  - these following parameters were fixed:

    ``` bash
    --meta 1.35 --lr 0.005 --decay 1e-7 --epochs-per-task 20
    --task-sequence 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST' 'pMNIST'
    ```

  - data downloaded from `SynapticMetaplasticityBNN/Continual_Learning_Fig-2abcdefgh-3abcd-5cde/results/2021-11-04` of [V2]

[paper-repo]: https://github.com/Laborieux-Axel/SynapticMetaplasticityBNN
[paper]: https://www.nature.com/articles/s41467-021-22768-y
[source]: https://www.kaggle.com/penguinsfly/bnn-meta-rerun-from-paper
[V1]: https://www.kaggle.com/penguinsfly/bnn-meta-rerun-from-paper/data?scriptVersionId=78778974
[V2]: https://www.kaggle.com/penguinsfly/bnn-meta-rerun-from-paper/data?scriptVersionId=78817005
