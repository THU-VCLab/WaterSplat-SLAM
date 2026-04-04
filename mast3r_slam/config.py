import re
import warnings
import yaml

config = {}


def load_config(path="config/base.yaml", is_parent=False):
    # from https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(
            """^(?:
                [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
                |[-+]?\\.(?:inf|Inf|INF)
                |\\.(?:nan|NaN|NAN))$""",
            re.X,
        ),
        list("-+0123456789."),
    )

    with open(path, "r") as f:
        cfg = yaml.load(f, Loader=loader)
    inherit = cfg.get("inherit")
    if inherit is not None:
        cfg_parent = load_config(inherit, is_parent=True)
    else:
        cfg_parent = dict()
    cfg = merge_config(cfg_parent, cfg)
    if is_parent:
        return cfg

    _migrate_config(cfg)
    # update the global config only once at the child node
    set_global_config(cfg)


def _migrate_config(cfg):
    """Migrate legacy config keys to the new unified format.

    Handles:
    - Uppercase section names (Dataset, Training, Results) -> lowercase
    - Scattered output paths (outputs, save_dir, eval_traj) -> output section
    """
    # Normalize uppercase section names to lowercase
    _rename_map = {"Dataset": "dataset", "Training": "training", "Results": "results"}
    for old_key, new_key in _rename_map.items():
        if old_key in cfg:
            if new_key not in cfg:
                cfg[new_key] = {}
            merge_config(cfg[new_key], cfg.pop(old_key))
            warnings.warn(
                f"Config section '{old_key}' is deprecated, use '{new_key}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )

    # Unify output paths into output section
    if "output" not in cfg:
        cfg["output"] = {}
    output = cfg["output"]

    if "save_dir" in cfg and "base_dir" not in output:
        output["base_dir"] = cfg.pop("save_dir")
        warnings.warn(
            "Top-level 'save_dir' is deprecated, use 'output.base_dir' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    elif "save_dir" in cfg:
        cfg.pop("save_dir")

    if "outputs" in cfg and "gs_dir" not in output:
        output["gs_dir"] = cfg.pop("outputs")
        warnings.warn(
            "Top-level 'outputs' is deprecated, use 'output.gs_dir' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
    elif "outputs" in cfg:
        cfg.pop("outputs")

    if "eval_traj" in cfg and "eval_traj" not in output:
        output["eval_traj"] = cfg.pop("eval_traj")
    elif "eval_traj" in cfg:
        cfg.pop("eval_traj")

    # Remove dead config
    if "results" in cfg and "save_dir" in cfg["results"]:
        cfg["results"].pop("save_dir")

    # Ensure defaults
    output.setdefault("base_dir", "outputs/default")
    output.setdefault("gs_dir", "./outputs")
    output.setdefault("eval_traj", True)


def merge_config(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            merge_config(dict1[k], v)
        else:
            dict1[k] = v
    return dict1


def set_global_config(cfg):
    global config
    config.update(cfg)
    return config
