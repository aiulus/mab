import os


def ensure_output_dirs(dirs=None):
    if dirs is None:
        dirs = ["outputs/plots", "outputs/logs", "outputs/data"]
    for path in dirs:
        os.makedirs(path, exist_ok=True)
