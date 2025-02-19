from typing import Any

import flaml.tune
import flaml.tune.sample
import hyperopt.pyll
from hyperopt import hp
from hyperopt.pyll import scope
import optuna


def add_prefix(prefix: str, name: str) -> str:
    return f"{prefix}_{name}" if prefix else name


def convert_to_hyperopt(param_cfg: Any, prefix: str = '', name: str = 'name') -> Any:
    if not isinstance(param_cfg, dict) and not isinstance(param_cfg, list):
        return param_cfg

    param_cfg = param_cfg.copy()

    if isinstance(param_cfg, dict):
        new_config = dict()
        if name in param_cfg.keys():
            name_value = param_cfg.pop(name)
            return {
                name: name_value,
                **convert_to_hyperopt(param_cfg, add_prefix(prefix, name_value), name)
            }

        for k, v in param_cfg.items():
            if isinstance(v, dict):
                if 'args' in v.keys():
                    new_config[k] = get_hyperopt_sampler(v['args'], v['sampler'], add_prefix(prefix, k))
                else:
                    new_config[k] = convert_to_hyperopt(v, add_prefix(prefix, k), name)
            else:
                new_config[k] = convert_to_hyperopt(v, add_prefix(prefix, k), name)
        return new_config
    elif isinstance(param_cfg, list):
        new_config = list()
        for item in param_cfg:
            new_config.append(convert_to_hyperopt(item, prefix, name))
        return hp.choice(prefix + "_" + 'name', new_config)

    else:
        return param_cfg


def convert_to_optuna(trial: optuna.Trial, param_cfg: Any, prefix: str = '', name: str = 'name') -> dict:
    param_cfg = param_cfg.copy()

    out = dict()

    if isinstance(param_cfg, dict):
        if name in param_cfg.keys():
            name_value = param_cfg.pop(name)
            return {
                "name": name_value,
                **convert_to_optuna(trial, param_cfg, add_prefix(prefix, name_value), name)
            }

    for k, v in param_cfg.items():
        if isinstance(v, dict):
            if 'args' in v.keys():
                out[k] = get_optuna_sampler(v['args'], v['sampler'], add_prefix(prefix, k), trial)
            else:
                out[k] = convert_to_optuna(trial, v, add_prefix(prefix, k), name)
        elif isinstance(v, list):
            selected = trial.suggest_categorical(add_prefix(prefix, k), v)
            out[k] = convert_to_optuna(trial, selected, add_prefix(prefix, k), name)
        else:
            out[k] = v

    return out


def convert_to_flaml(param_cfg: dict | list) -> Any:
    param_cfg = param_cfg.copy()

    if isinstance(param_cfg, dict):
        new_config = dict()
        if 'name' in param_cfg.keys():
            name = param_cfg.pop('name')
            return {
                "name": name,
                **convert_to_flaml(param_cfg)
            }

        for k, v in param_cfg.items():
            if isinstance(v, dict):
                if 'args' in v.keys():
                    new_config[k] = get_flaml_sampler(v['args'], v['sampler'])
                else:
                    new_config[k] = convert_to_flaml(v)
            else:
                new_config[k] = convert_to_flaml(v)
        return new_config
    elif isinstance(param_cfg, list):
        new_config = list()
        for item in param_cfg:
            new_config.append(convert_to_flaml(item))
        return flaml.tune.choice(new_config)

    else:
        return param_cfg


def get_sampler(
        package_name: str,
        arg: list,
        sampler: str,
        sample_name: str = "",
        trial: optuna.Trial = None) -> Any:
    match package_name:
        case "flaml":
            return get_flaml_sampler(arg, sampler)
        case "hyperopt":
            return get_hyperopt_sampler(arg, sampler, sample_name)
        case "optuna":
            return get_optuna_sampler(arg, sampler, sample_name, trial)
        case _:
            raise ValueError(f"Package {package_name} not supported")


def get_flaml_sampler(arg: list, sampler: str) -> flaml.tune.sample.Domain:
    match sampler:
        case "uniform":
            return flaml.tune.uniform(*arg)
        case "loguniform":
            return flaml.tune.loguniform(*arg)
        case "quniform":
            return flaml.tune.quniform(*arg)
        case "qloguniform":
            return flaml.tune.qloguniform(*arg)
        case "uniformint":
            return flaml.tune.randint(*arg)
        case "quniformint":
            return flaml.tune.qrandint(*arg)
        case "loguniformint":
            return flaml.tune.lograndint(*arg)
        case "qloguniformint":
            return flaml.tune.qlograndint(*arg)
        case "choice":
            return flaml.tune.choice(arg)
        case _:
            raise ValueError(f"Sampler {sampler} not supported")


def get_hyperopt_sampler(arg: list, sampler: str, sample_name: str) -> hyperopt.pyll.Apply:
    match sampler:
        case "uniform":
            return hp.uniform(sample_name, *arg)
        case "loguniform":
            return hp.loguniform(sample_name, *arg)
        case "quniform":
            return hp.quniform(sample_name, *arg)
        case "qloguniform":
            return hp.qloguniform(sample_name, *arg)
        case "uniformint":
            return hp.randint(sample_name, *arg)
        case "quniformint":
            return scope.int(hp.quniform(sample_name, *arg))
        case "loguniformint":
            return scope.int(hp.loguniform(sample_name, *arg))
        case "qloguniformint":
            return scope.int(hp.qloguniform(sample_name, *arg))
        case "choice":
            return hp.choice(sample_name, arg)
        case _:
            raise ValueError(f"Sampler {sampler} not supported")


def get_optuna_sampler(arg: list,
                       sampler: str,
                       sample_name: str,
                       trial: optuna.Trial) -> Any:
    match sampler:
        case "uniform":
            return trial.suggest_float(sample_name, *arg)
        case "loguniform":
            return trial.suggest_float(sample_name, *arg, log=True)
        case "quniform":
            return trial.suggest_float(sample_name, arg[0], arg[1], step=arg[2])
        case "qloguniform":
            return trial.suggest_float(sample_name, arg[0], arg[1], step=arg[2], log=True)
        case "uniformint":
            return trial.suggest_int(sample_name, *arg)
        case "quniformint":
            return trial.suggest_int(sample_name, arg[0], arg[1], step=arg[2])
        case "loguniformint":
            return trial.suggest_int(sample_name, arg[0], arg[1], log=True)
        case "qloguniformint":
            return trial.suggest_int(sample_name, arg[0], arg[1], step=arg[2], log=True)
        case "choice":
            return trial.suggest_categorical(sample_name, arg)
        case _:
            raise ValueError(f"Sampler {sampler} not supported")


def get_estimator_class(estimator_name: str) -> Any:
    package = estimator_name.rsplit(".", 1)[0]
    try:
        exec(f"import {package}")
    except ImportError:
        return None
    return eval(estimator_name)


def _match_flaml_domain(domain: flaml.tune.sample.Domain) -> dict:
    """
    Match the FLAML domain to the definition used in this library.
    """
    domain_type = domain.__class__.__name__
    if domain_type == 'Categorical':
        return {"args": domain.categories, "sampler": "choice"}

    lower = domain.lower
    upper = domain.upper
    args = [lower, upper]

    sampler = domain.get_sampler()
    sampler_name = ''

    if sampler.__class__.__name__ == 'Quantized':
        sampler_name += 'q'
        sampler = sampler.sampler
        args += [sampler.q]

    sampler_name += sampler.__class__.__name__.lower()[1:]

    if domain_type == 'Integer':
        sampler_name += 'int'

    return {"args": args, "sampler": sampler_name}


def _transform_hyperopt(config: dict) -> dict:
    """
    Transform the configuration from Hyperopt format to the format used in this library.
    """
    # TODO: Implement this
    pass


def _transform_flaml(config: dict) -> dict:
    """
    Transform the configuration from FLAML format to the format used in this library.
    """
    new_config = dict()
    for k, v in config.items():
        if isinstance(v, dict):
            new_config[k] = _transform_flaml(v)
        elif isinstance(v, flaml.tune.sample.Domain):
            new_config[k] = _match_flaml_domain(v)
        else:
            new_config[k] = v

    return new_config

