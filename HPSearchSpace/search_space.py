from .utils import convert_to_hyperopt_space, suggest_classifier, get_flaml_sampler, get_estimator_class

from typing import Self, SupportsFloat

import yaml
import flaml.tune
from optuna import Trial


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

    def get_hyperopt_space(self) -> dict:
        """
        :return: A dictionary that defines the search space for hyperopt.
        """
        return convert_to_hyperopt_space(self.config)

    def get_optuna_space(self, trial_: Trial) -> dict:
        return suggest_classifier(trial_, self.config)

    def get_flaml_space(self) -> dict:
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
        for key, estimators in self.config.items():
            if key not in estimator_list.keys():
                del self.config[key]
                continue
            self.config[key] = {k: v for k, v in estimators.items() if k in estimator_list[key]}
        return self

    def join(self, other_search_space: Self) -> Self:
        self.config = {**self.config, **other_search_space.config}
        return self
