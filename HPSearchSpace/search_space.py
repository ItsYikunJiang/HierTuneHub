import yaml


class SearchSpace:
    """
    The class for defining the search space for hyperparameter optimization.
    """
    def __init__(self,
                 config: dict = None,
                 config_file: str = None):
        """
        Initialize the search space. You can either provide the configuration as a dictionary or as a YAML file.
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

