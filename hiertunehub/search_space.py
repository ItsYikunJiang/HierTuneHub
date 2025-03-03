from .utils import (
    convert_to_hyperopt, convert_to_optuna, convert_to_flaml,
    _transform_flaml, _transform_hyperopt
)

import sys
if sys.version_info >= (3, 11):
    from typing import Self, Any, TYPE_CHECKING
else:
    from typing import Any as Self
    from typing import Any, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import optuna


class SearchSpace:
    """
    The class for defining the search space for hyperparameter optimization.
    """

    def __init__(self,
                 config_path: str = None,
                 name: str = 'name',
                 sep: str = '?'
                 ):
        """
        Initialize the search space. You can either provide the configuration as a dictionary or as a YAML file.
        The input configuration should resemble certain structure. See example.yaml for an example.
        :param config_path: A YAML file containing the configuration for the search space.
        :param name: The unique name key in the search space that will be used to identify different hyperparameters.
        :param sep: The separator used to join the name key with the hyperparameter
        """
        if config_path is not None:
            with open(config_path, 'r') as stream:
                self.config = yaml.safe_load(stream)
                self.config = self._parse_config(self.config)
        else:
            self.config = None

        self.name = name
        self.sep = sep

    @classmethod
    def from_dict(cls, config: dict, name: str = 'name', sep: str = '?') -> Self:
        """
        Initialize the search space from a dictionary resembling the YAML configuration.
        :param config: A dictionary containing the configuration for the search space.
        :param name: The unique name key in the search space that will be used to identify different hyperparameters.
        :param sep: The separator used to join the name key with the hyperparameter
        :return: SearchSpace object
        """
        search_space = cls(name=name, sep=sep)
        search_space.config = config
        search_space.config = search_space._parse_config(search_space.config)
        return search_space

    @classmethod
    def from_flaml(cls, config: dict, name: str = 'name', sep: str = '?') -> Self:
        """
        Initialize the search space from a FLAML configuration.
        :param config: A dictionary containing the configuration for the search space in FLAML format.
        :param name: The unique name key in the search space that will be used to identify different hyperparameters.
        :param sep: The separator used to join the name key with the hyperparameter
        :return: SearchSpace object
        """
        search_space = cls(name=name, sep=sep)
        search_space.config = _transform_flaml(config)
        search_space.config = search_space._parse_config(search_space.config)
        return search_space

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
        return convert_to_hyperopt(self.config, name=self.name, sep=self.sep)

    def to_optuna(self, trial: 'optuna.Trial') -> dict:
        """
        :param trial: An optuna trial object.
        :return: A dictionary that outputs a sample from the search space.
        """
        return convert_to_optuna(trial, self.config, name=self.name, sep=self.sep)

    def to_flaml(self) -> dict:
        """
        :return: A dictionary that defines the search space for FLAML.
        """
        return convert_to_flaml(self.config, name=self.name)

    def join(self, other_search_space: Self) -> Self:
        """
        Join the current search space with another distinct search space.
        :param other_search_space: Another search space object.
        :return: self
        """
        self.config.update(other_search_space.config)
        return self


