from typing import Self, SupportsFloat, Any

import yaml
import flaml


class SearchSpace:
    """
    The class for defining the search space for hyperparameter optimization.
    """
    def __init__(self,
                 config: dict[str, dict[str, dict[str, dict[str, list | str | SupportsFloat]]]] = None,
                 config_file: str = None):
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

    def get_hyperopt_space(self) -> dict:
        """
        :return: A dictionary that defines the search space for hyperopt.
        """
        ...

    def get_optuna_space(self):
        # TODO: How to implement this? Optuna does not explicitly define a search space.
        ...

    def get_flaml_space(self) -> dict:
        config = self.config.copy()

        out = dict()

        for key, estimators in config.items():
            space = list()

            for estimator_name, params in estimators.items():
                params_space = dict()

                for params_key, params_dict in params.items():
                    if "values" in params_dict.keys():
                        params_space[params_key] = flaml.tune.choice(params_dict["values"])
                    elif "range" in params_dict.keys():
                        params_space[params_key] = self._get_flaml_range(params_dict["range"], params_dict["sampler"])

                single_space = {**params_space, "estimator_name": estimator_name,
                                "estimator_class": self._get_estimator_class(estimator_name)}
                space.append(single_space)

            out[key] = flaml.tune.choice(space)
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

    @staticmethod
    def _get_flaml_range(range_, sampler):
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
            case _:
                raise ValueError(f"Sampler {sampler} not supported")

    @staticmethod
    def _get_estimator_class(estimator_name: str) -> Any:
        package = estimator_name.rsplit(".", 1)[0]
        exec(f"import {package}")
        return eval(estimator_name)

