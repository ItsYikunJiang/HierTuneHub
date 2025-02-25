from .utils import (
    convert_to_hyperopt, convert_to_optuna, convert_to_flaml,
    _transform_flaml, _transform_hyperopt
)

import sys
if sys.version_info >= (3, 11):
    from typing import Self, Union, Any
else:
    from typing import Any as Self
    from typing import Union, Any

import yaml
import optuna


class SearchSpace:
    """
    The class for defining the search space for hyperparameter optimization.
    """

    def __init__(self,
                 config_file: str = None,
                 config: Union[dict, list] = None,
                 config_framework: str = None,
                 name: str = 'name'
                 ):
        """
        Initialize the search space. You can either provide the configuration as a dictionary or as a YAML file.
        The input configuration should resemble certain structure. See example.yaml for an example.
        :param config: A dictionary containing the configuration for the search space.
        :param config_file: A YAML file containing the configuration for the search space.
        :param config_framework: If provided, you should provide config dict which is in the format of the
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
            if config_framework == 'flaml':
                self.config = _transform_flaml(self.config)
            elif config_framework == 'hyperopt':
                self.config = _transform_hyperopt(self.config)
            else:
                raise ValueError(f"Config type {config_framework} not supported")

        self.name = name

    @staticmethod
    def _parse_config(config: Any) -> Any:
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
        return convert_to_hyperopt(self.config, name=self.name)

    def to_optuna(self, trial: optuna.Trial) -> dict:
        """
        :param trial: An optuna trial object.
        :return: A dictionary that outputs a sample from the search space.
        """
        return convert_to_optuna(trial, self.config, name=self.name)

    def to_flaml(self) -> dict:
        """
        :return: A dictionary that defines the search space for FLAML.
        """
        return convert_to_flaml(self.config)

    def join(self, other_search_space: Self) -> Self:
        """
        Join the current search space with another distinct search space.
        :param other_search_space: Another search space object.
        :return: self
        """
        self.config.update(other_search_space.config)
        return self


