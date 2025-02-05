from typing import Any

import flaml.tune
from hyperopt import hp
from hyperopt.pyll import scope
from optuna import Trial


def convert_to_hyperopt_space(param_dict: dict) -> dict:
    param_dict = param_dict.copy()

    out = dict()

    for estimator_group_name, estimators_dict in param_dict.items():
        space = list()

        for estimator_name, params_dict in estimators_dict.items():
            params_space = dict()

            for params_key, params_config in params_dict.items():
                params_space[params_key] = get_hyperopt_sampler(
                    params_config['args'], params_config["sampler"],
                    estimator_group_name + "_" + estimator_name + "_" + params_key
                )

            single_space = {"params": params_space, "estimator_name": estimator_name,
                            "estimator_class": get_estimator_class(estimator_name)}
            space.append(single_space)

        out[estimator_group_name] = hp.choice(estimator_group_name, space)

    return out


def suggest_classifier(trial_: Trial, param_config: dict) -> dict:
    param_config = param_config.copy()

    out = dict()

    for estimator_group_name, estimators_dict in param_config.items():
        suggested_estimator_name = trial_.suggest_categorical(estimator_group_name, list(estimators_dict.keys()))
        suggested_estimator_class = get_estimator_class(suggested_estimator_name)

        params_space = dict()
        for params_key, params_config in estimators_dict[suggested_estimator_name].items():
            params_space[params_key] = get_optuna_sampler(
                params_config['args'], params_config["sampler"],
                estimator_group_name + "_" + suggested_estimator_name + "_" + params_key,
                trial_
            )

        out[estimator_group_name] = {
            "params": params_space,
            "estimator_name": suggested_estimator_name,
            "estimator_class": suggested_estimator_class
        }

    return out


def get_sampler(
        package_name: str,
        range_: list,
        sampler: str,
        sample_name: str = "",
        trial_=None) -> Any:
    match package_name:
        case "flaml":
            return get_flaml_sampler(range_, sampler)
        case "hyperopt":
            return get_hyperopt_sampler(range_, sampler, sample_name)
        case "optuna":
            return get_optuna_sampler(range_, sampler, sample_name, trial_)
        case _:
            raise ValueError(f"Package {package_name} not supported")


def get_flaml_sampler(range_: list, sampler: str) -> Any:
    match sampler:
        case "uniform":
            return flaml.tune.uniform(*range_)
        case "loguniform":
            return flaml.tune.loguniform(*range_)
        case "quniform":
            return flaml.tune.quniform(*range_)
        case "qloguniform":
            return flaml.tune.qloguniform(*range_)
        case "uniformint":
            return flaml.tune.randint(*range_)
        case "quniformint":
            return flaml.tune.qrandint(*range_)
        case "loguniformint":
            return flaml.tune.lograndint(*range_)
        case "qloguniformint":
            return flaml.tune.qlograndint(*range_)
        case "choice":
            return flaml.tune.choice(range_)
        case _:
            raise ValueError(f"Sampler {sampler} not supported")


def get_hyperopt_sampler(range_: list, sampler: str, sample_name: str) -> Any:
    match sampler:
        case "uniform":
            return hp.uniform(sample_name, *range_)
        case "loguniform":
            return hp.loguniform(sample_name, *range_)
        case "quniform":
            return hp.quniform(sample_name, *range_)
        case "qloguniform":
            return hp.qloguniform(sample_name, *range_)
        case "uniformint":
            return hp.randint(sample_name, *range_)
        case "quniformint":
            return scope.int(hp.quniform(sample_name, *range_))
        case "loguniformint":
            return scope.int(hp.loguniform(sample_name, *range_))
        case "qloguniformint":
            return scope.int(hp.qloguniform(sample_name, *range_))
        case "choice":
            return hp.choice(sample_name, range_)
        case _:
            raise ValueError(f"Sampler {sampler} not supported")


def get_optuna_sampler(range_: list,
                       sampler: str,
                       sample_name: str,
                       trial_) -> Any:
    match sampler:
        case "uniform":
            return trial_.suggest_float(sample_name, *range_)
        case "loguniform":
            return trial_.suggest_float(sample_name, *range_, log=True)
        case "quniform":
            return trial_.suggest_float(sample_name, range_[0], range_[1], step=range_[2])
        case "qloguniform":
            return trial_.suggest_float(sample_name, range_[0], range_[1], step=range_[2], log=True)
        case "uniformint":
            return trial_.suggest_int(sample_name, *range_)
        case "quniformint":
            return trial_.suggest_int(sample_name, range_[0], range_[1], step=range_[2])
        case "loguniformint":
            return trial_.suggest_int(sample_name, range_[0], range_[1], log=True)
        case "qloguniformint":
            return trial_.suggest_int(sample_name, range_[0], range_[1], step=range_[2], log=True)
        case "choice":
            return trial_.suggest_categorical(sample_name, range_)
        case _:
            raise ValueError(f"Sampler {sampler} not supported")


def get_estimator_class(estimator_name: str) -> Any:
    package = estimator_name.rsplit(".", 1)[0]
    try:
        exec(f"import {package}")
    except ImportError:
        return None
    return eval(estimator_name)
