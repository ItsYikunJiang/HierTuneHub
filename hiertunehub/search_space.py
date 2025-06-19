from .utils import (
    convert_to_hyperopt, convert_to_optuna, convert_to_flaml,
    _transform_flaml, _transform_hyperopt, add_prefix
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
    import optuna.distributions as op_dis


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

    def point_to_hyperopt_representation(self,
                                         config: dict,
                                         ) -> dict:
        """
        Convert a sampled point in the search space to a hyperopt representation.
        :param config: A dictionary representing a sampled point in the search space.
        :return: A dictionary that represents the sampled params in hyperopt format.
        """
        return self._point_to_hyperopt_representation(config)

    def _point_to_hyperopt_representation(self,
                                          config: dict,
                                          prefix: str = '',
                                          ss: dict = None,
                                          ) -> dict:
        if ss is None:
            ss = self.config
        out = dict()
        for k, v in config.items():
            prefix_k = add_prefix(prefix, k, self.sep)

            if isinstance(v, dict):
                if self.name in v.keys():
                    name_val = v.pop(self.name)
                    name_key = add_prefix(prefix_k, name_val, self.sep)
                    prefix_sep = prefix_k.split(self.sep)
                    for pref in prefix_sep:
                        if pref not in ss:
                            pass
                        else:
                            ss = ss[pref]
                    # Now ss should be a list of dict
                    for i, s in enumerate(ss):
                        if s[self.name] == name_val:
                            out[add_prefix(prefix_k, self.name, self.sep)] = i
                            break
                    if v:
                        out.update(self._point_to_hyperopt_representation(v, prefix=name_key, ss=s))
            else:
                if ss[k]['sampler'] == 'choice':
                    for i, value in enumerate(ss[k]['args']):
                        if value == v:
                            out[prefix_k] = i
                            break
                else:
                    out[prefix_k] = v

        return out

    def point_to_optuna_representation(self,
                                       config: dict) -> tuple[dict, dict]:
        """
        Convert a sampled point in the search space to an optuna representation.
        :param config: A dictionary representing a sampled point in the search space.
        :return: A tuple containing the params dict and distributions dict for optuna.
        """
        return self._point_to_optuna_representation(config)

    def _point_to_optuna_representation(self,
                                        config: dict,
                                        prefix: str = '',
                                        ss: dict = None,
                                        ) -> tuple[dict, dict]:
        global op_dis
        import optuna.distributions as op_dis
        if ss is None:
            ss = self.config
        out_p = dict()
        out_d = dict()
        for k, v in config.items():
            prefix_k = add_prefix(prefix, k, self.sep)

            if isinstance(v, dict):
                if self.name in v.keys():
                    name_val = v.get(self.name)
                    name_key = add_prefix(prefix_k, name_val, self.sep)
                    prefix_sep = prefix_k.split(self.sep)
                    for pref in prefix_sep:
                        if pref not in ss:
                            pass
                        else:
                            ss = ss[pref]
                    # Now ss should be a list of dict
                    out_d[prefix_k] = op_dis.CategoricalDistribution(
                        choices=ss
                    )
                    for i, s in enumerate(ss):
                        if s[self.name] == name_val:
                            out_p[prefix_k] = s
                            break
                    if v:
                        v_out_p, v_out_d = self._point_to_optuna_representation(v, prefix=name_key, ss=s)
                        out_p.update(v_out_p)
                        out_d.update(v_out_d)
            elif k == self.name:
                pass
            else:
                out_p[prefix_k] = v
                out_d[prefix_k] = _get_optuna_distribution(ss[k])

        return out_p, out_d


def _get_optuna_distribution(args_and_sampler: dict) -> 'op_dis.BaseDistribution':
    global op_dis
    import optuna.distributions as op_dis
    arg = args_and_sampler['args']
    sampler = args_and_sampler['sampler']
    if sampler == "uniform":
        return op_dis.FloatDistribution(low=arg[0], high=arg[1])
    elif sampler == "loguniform":
        return op_dis.FloatDistribution(low=arg[0], high=arg[1], log=True)
    elif sampler == "quniform":
        return op_dis.FloatDistribution(low=arg[0], high=arg[1], step=arg[2])
    elif sampler == "qloguniform":
        return op_dis.FloatDistribution(low=arg[0], high=arg[1], log=True)
    elif sampler == "uniformint":
        return op_dis.IntDistribution(low=arg[0], high=arg[1])
    elif sampler == "quniformint":
        return op_dis.IntDistribution(low=arg[0], high=arg[1], step=arg[2])
    elif sampler == "loguniformint":
        return op_dis.IntDistribution(low=arg[0], high=arg[1], log=True)
    elif sampler == "qloguniformint":
        return op_dis.IntDistribution(low=arg[0], high=arg[1], log=True)
    elif sampler == "choice":
        return op_dis.CategoricalDistribution(choices=arg)
    else:
        raise ValueError(f"Unknown sampler {sampler} for optuna distribution.")
