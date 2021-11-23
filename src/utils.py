from pathlib import Path

"""Contains util functions."""


def create_dir_recursive(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return


if __name__ == '__main__':
    pass
