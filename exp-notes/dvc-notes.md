# General notes on workflows

These are **experimental**.

## DVC set up

### DVC general

Assuming things are set up for pulling and pushing already (see the **Set up** sections):

- `dvc add data/input/<DIRECTORY-DATA>` -> will generate a file called `data/input/<DIRECTORY-DATA>.dvc` (remember to `.gitkeep` the `*.dvc` files)
- then `dvc push` + `git add` and `git commit`
- on another machine, do `git pull` and then `dvc pull`

### Set up on local machine

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

- Set up remote for `dvc`
  - First set up [google cloud remote](https://dvc.org/doc/user-guide/setup-google-drive-remote#using-a-custom-google-cloud-project-recommended)
    - recommended if using google drive
    - only need to do once, ask whoever did it first
  - Then (`--local` tag so not tracked by `git`)

    ``` bash
    dvc add <FILES OR DIR> # add some thing

    # Now add remote named "data-storage"
    dvc remote add --default --local data-storage gdrive://<URL>
    dvc remote modify --local data-storage gdrive_client_id 'client_id'
    dvc remote modify --local data-storage gdrive_client_secret 'client_secret'

    dvc push # this will lead to the token auth link
    ```

  - In case want to revoke 3rd party access later, follow [here]( https://support.google.com/accounts/answer/3466521?hl=en)

### Set up on remote machine

On each remote (i.e. machine):

- Clone first `git clone https://github.com/tuanpham96/bnn-cf-vs-robust`
- Optional: Set up ssh key on each remote if doesn't want to use password:
  - If use general ssh key: follow [this](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
  - If use deploy key, see [here](https://docs.github.com/en/developers/overview/managing-deploy-keys#deploy-keys)
    - reason: because I only want push privilege to only within this repo from this remote
    - on remote, [set up](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) key with `ssh-keygen`
    - copy remote's `~/ssh/<key-name>.pub` content to github repo `Settings/Deploy keys` section
    - then on remote (parts from [here](https://gist.github.com/xirixiz/b6b0c6f4917ce17a90e00f9b60566278))

      ``` bash
      # test the SSH key
      ssh -T git@github.com

      # user config (--global tag optional)
      git config user.email <EMAIL>
      git config user.name <NAME>

      # remote url
      git remote set-url origin git@github.com:tuanpham96/bnn-cf-vs-robust.git
      ```

  - after this `git push` is available
- Install `dvc`: `pip install 'dvc[gdrive]'`
- Optional:

  ``` bash
  # completion
  mkdir -p  ~/.local/share/bash-completion/
  dvc completion -s bash >  ~/.local/share/bash-completion/dvc
  source ~/.local/share/bash-completion/dvc # need to run each time
  # remove analytics
    dvc config --global core.analytics false
    rm ~/.config/dvc/user_id*
  ```

- Set up `dvc` remote:

  ``` bash
  dvc remote add --default --local data-storage gdrive://<URL>
  dvc remote modify --local data-storage gdrive_client_id 'client_id'
  dvc remote modify --local data-storage gdrive_client_secret 'client_secret'
  ```

- From local machine, just copy over the file `.dvc/tmp/gdrive-user-credentials.json` content to the appropriate repo directory on remote, don't need another token
- Then `dvc pull`

[dvc-install]: https://dvc.org/doc/install/linux
[dvc-completion]: https://dvc.org/doc/install/completion
[dvc-analytics]: https://dvc.org/doc/user-guide/analytics