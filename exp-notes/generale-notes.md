# General notes on workflows

- In an environment on local machine or on remote (i.e. Colab or specialized machine), [install][dvc-install] `dvc` by:

  ``` bash
  pip install "dvc[all]"
  ```

  - Either use `pip`, `conda` or even `pipx` to avoid clashing with current environment dependencies.
  - If only need S3 or google drive storage, just use `"dvc[s3]"` so don't have to install everything, see [dvc-install].
  - (Optional) Add bash [completion][dvc-completion] by doing:

    ``` bash
    dvc completion -s bash | tee ~/.local/share/bash-completion/completions/dvc
    ```

  - (Optional but rec) Turn off the [creepiness][dvc-analytics]:

    ``` bash
    dvc config --global core.analytics false
    dvc config --system core.analytics false # if accidentally install with sudo
    rm ~/.config/dvc/user_id*
    ```

[dvc-install]: https://dvc.org/doc/install/linux
[dvc-completion]: https://dvc.org/doc/install/completion
[dvc-analytics]: https://dvc.org/doc/user-guide/analytics