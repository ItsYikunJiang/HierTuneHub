from .utils import convert_to_hyperopt_space, suggest_classifier, get_flaml_sampler, get_estimator_class

from typing import Self, SupportsFloat

import yaml
import flaml.tune
import optuna

# TODO: support arbitrary number of hierarchies in the configuration


class SearchSpace:
    """
    The class for defining the search space for hyperparameter optimization.
    """

    def __init__(self,
                 config_file: str = None,
                 config: dict[str, dict[str, dict[str, dict[str, list | str | SupportsFloat]]]] = None,
                 ):
        """
        Initialize the search space. You can either provide the configuration as a dictionary or as a YAML file.
        The input configuration should resemble certain structure. See example.yaml for an example.
        :param config: A dictionary containing the configuration for the search space.
        :param config_file: A YAML file containing the configuration for the search space.
        """
        if config is None and config_file is None:
            raise ValueError("Either config or config_file must be provided")
        if config is not None and config_file is not None:
            raise ValueError("Only one of config or config_file must be provided")

        if config is not None:
            self.config = config
        elif config_file is not None:
            with open(config_file, 'r') as stream:
                self.config = yaml.safe_load(stream)

        self._parse_config()

    def _parse_config(self):
        """
        Parse the configuration to map values and range to args for simplicity.
        """
        for estimator_group_name, estimators_dict in self.config.items():
            for estimator_name, params_dict in estimators_dict.items():
                for params_name, params_config in params_dict.items():
                    if "values" in params_config.keys() and "range" in params_config.keys():
                        raise ValueError("Both values and range cannot be provided for a parameter")
                    if "values" not in params_config.keys() and "range" not in params_config.keys():
                        raise ValueError("Either values or range must be provided for a parameter")

                    if "values" in params_config.keys():
                        params_config['args'] = params_config.pop('values')
                        params_config['sampler'] = "choice"
                    elif "range" in params_config.keys():
                        params_config['args'] = params_config.pop('range')

    def to_hyperopt(self) -> dict:
        """
        :return: A dictionary that defines the search space for hyperopt.
        """
        return convert_to_hyperopt_space(self.config)

    def to_optuna(self, trial: optuna.Trial) -> dict:
        """
        :param trial: An optuna trial object.
        :return: A dictionary that outputs a sample from the search space.
        """
        return suggest_classifier(trial, self.config)

    def to_flaml(self) -> dict:
        """
        :return: A dictionary that defines the search space for FLAML.
        """
        config = self.config.copy()

        out = dict()

        for estimator_group_name, estimators_dict in config.items():
            space = list()

            for estimator_name, params_dict in estimators_dict.items():
                params_space = dict()

                for params_key, params_config in params_dict.items():
                    params_space[params_key] = get_flaml_sampler(params_config['args'], params_config["sampler"])

                single_space = {"params": params_space, "estimator_name": estimator_name,
                                "estimator_class": get_estimator_class(estimator_name)}
                space.append(single_space)

            out[estimator_group_name] = flaml.tune.choice(space)
        return out

    def select(self, estimator_list: dict[str, list]) -> Self:
        """
        Select a subset of the search space based on the provided estimator_list.
        :param estimator_list: A dictionary with keys as estimator group names and values as list of estimator names
        that should be included in the search space.
        :return: self
        """
        for estimator_group_name, estimators_dict in self.config.items():
            if estimator_group_name not in estimator_list.keys():
                del self.config[estimator_group_name]
                continue
            self.config[estimator_group_name] = {
                k: v for k, v in estimators_dict.items() if k in estimator_list[estimator_group_name]
            }
        return self

    def join(self, other_search_space: Self) -> Self:
        """
        Join the current search space with another search space.
        :param other_search_space: Another search space object.
        :return: self
        """
        for estimator_group_name, estimators_dict in other_search_space.config.items():
            if estimator_group_name not in self.config.keys():
                self.config[estimator_group_name] = estimators_dict
            else:
                for estimator_name, params_dict in estimators_dict.items():
                    if estimator_name not in self.config[estimator_group_name].keys():
                        self.config[estimator_group_name][estimator_name] = params_dict
                    else:
                        self.config[estimator_group_name][estimator_name].update(params_dict)
        return self
