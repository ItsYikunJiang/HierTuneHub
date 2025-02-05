from typing import Any

import flaml.tune
import flaml.tune.sample
import hyperopt.pyll
from hyperopt import hp
from hyperopt.pyll import scope
import optuna


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


def suggest_classifier(trial: optuna.Trial, param_config: dict) -> dict:
    param_config = param_config.copy()

    out = dict()

    for estimator_group_name, estimators_dict in param_config.items():
        suggested_estimator_name = trial.suggest_categorical(estimator_group_name, list(estimators_dict.keys()))
        suggested_estimator_class = get_estimator_class(suggested_estimator_name)

        params_space = dict()
        for params_key, params_config in estimators_dict[suggested_estimator_name].items():
            params_space[params_key] = get_optuna_sampler(
                params_config['args'], params_config["sampler"],
                estimator_group_name + "_" + suggested_estimator_name + "_" + params_key,
                trial
            )

        out[estimator_group_name] = {
            "params": params_space,
            "estimator_name": suggested_estimator_name,
            "estimator_class": suggested_estimator_class
        }

    return out


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
