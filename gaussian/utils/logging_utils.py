import yaml
import numpy as np
import rich
import copy
import torch
from matplotlib import cm


_log_styles = {
    "GSBackend": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "PGBA": "bold blue",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"


def Log(*args, tag="GSBackend"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def clone_obj(obj):
    clone_obj = copy.deepcopy(obj)
    for attr in clone_obj.__dict__.keys():
        # check if its a property
        if hasattr(clone_obj.__class__, attr) and isinstance(
            getattr(clone_obj.__class__, attr), property
        ):
            continue
        if isinstance(getattr(clone_obj, attr), torch.Tensor):
            setattr(clone_obj, attr, getattr(clone_obj, attr).detach().clone())
    return clone_obj
