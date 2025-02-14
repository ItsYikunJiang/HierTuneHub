from .utils import convert_to_hyperopt, convert_to_optuna, convert_to_flaml

import numbers
from typing import Self, Union

import yaml
import flaml.tune
import optuna


class SearchSpace:
    """
    The class for defining the search space for hyperparameter optimization.
    """

    def __init__(self,
                 config_file: str = None,
                 config: Union[dict, list] = None,
                 config_framework: str = None
                 ):
        """
        Initialize the search space. You can either provide the configuration as a dictionary or as a YAML file.
        The input configuration should resemble certain structure. See example.yaml for an example.
        :param config: A dictionary containing the configuration for the search space.
        :param config_file: A YAML file containing the configuration for the search space.
        :param config_type: If provided, you should provide config dict which is in the format of the
        specified config type. Supported types are "flaml" and "hyperopt".
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

        if config_framework is None:
            self.config = self._parse_config(self.config)
        else:
            match config_framework:
                case "flaml":
                    self.config = self._transform_flaml(self.config)
                case "hyperopt":
                    self.config = self._transform_hyperopt(self.config)
                case _:
                    raise ValueError(f"Config type {config_framework} not supported")

    @staticmethod
    def _parse_config(config: dict | list | str) -> dict | list | str:
        """
        Parse the configuration to map values and range to args for simplicity.
        """

        if isinstance(config, dict):
            new_config = dict()

            if "values" in config.keys():
                new_config['args'] = config.pop('values')
                new_config['sampler'] = 'choice'
                config.pop('sampler', None)
                return new_config
            elif "range" in config.keys():
                new_config['args'] = config.pop('range')
                new_config['sampler'] = config.pop('sampler')
                return new_config

            for k, v in config.items():
                new_config[k] = SearchSpace._parse_config(v)

        elif isinstance(config, list):
            new_config = list()
            for item in config:
                new_config.append(SearchSpace._parse_config(item))

        else:
            new_config = config

        return new_config

    def to_hyperopt(self) -> dict:
        """
        :return: A dictionary that defines the search space for hyperopt.
        """
        return convert_to_hyperopt(self.config)

    def to_optuna(self, trial: optuna.Trial) -> dict:
        """
        :param trial: An optuna trial object.
        :return: A dictionary that outputs a sample from the search space.
        """
        return convert_to_optuna(trial, self.config)

    def to_flaml(self) -> dict:
        """
        :return: A dictionary that defines the search space for FLAML.
        """
        return convert_to_flaml(self.config)

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

    @staticmethod
    def _transform_flaml(config: dict) -> dict:
        """
        Transform the configuration from FLAML format to the format used in this library.
        """
        new_config = dict()
        for k, v in config.items():
            if isinstance(v, dict):
                new_config[k] = SearchSpace._transform_flaml(v)
            elif isinstance(v, flaml.tune.sample.Domain):
                new_config[k] = SearchSpace._match_flaml_domain(v)
            elif isinstance(v, (int, float, numbers.Number)):
                new_config[k] = v

        return new_config

    @staticmethod
    def _transform_hyperopt(config: dict) -> dict:
        """
        Transform the configuration from Hyperopt format to the format used in this library.
        """
        # TODO: Implement this
        pass

    @staticmethod
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


