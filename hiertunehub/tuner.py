from typing import Callable, Optional, Union, Any
from functools import wraps
from dataclasses import dataclass

from .search_space import SearchSpace
from .utils import add_prefix


@dataclass
class Trial:
    params: dict
    result: Union[float, dict]


class Tuner:
    def __init__(self,
                 objective: Callable,
                 search_space: SearchSpace,
                 mode: str = "min",
                 metric: Optional[str] = None,
                 framework: str = "hyperopt",
                 framework_params: dict = None,
                 **kwargs
                 ):
        """
        Initialize the tuner.
        :param objective: user-defined objective function. The objective function can ouput a single float value, or a
        dictionary containing the metric value and other values.
        :param search_space: A SearchSpace object containing the search space for the hyperparameters.
        :param mode: The optimization mode. Either 'min' or 'max'.
        :param metric: If the objective function returns a dictionary, the metric key specifies the key to be used for
        optimization.
        :param framework: The framework to be used for optimization. Supported frameworks are "hyperopt", "optuna", and
        "flaml".
        :param framework_params: Additional parameters to be passed to the framework tuning function.
        :param kwargs: Additional parameters to be passed to the framework tuning function.
        """
        self.objective = objective
        self.search_space = search_space

        self.mode = mode
        if self.mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'")

        self.framework = framework
        if framework_params is None:
            framework_params = {}
        self.framework_params = framework_params
        self.framework_params.update(kwargs)

        self.metric = metric

        self._trials = list()
        self.best_trial = None

    def wrap_objective(self, objective: Callable) -> Callable:
        """
        Wrap the objective function to return the result in the format expected by the framework.
        :param objective: The objective function to be wrapped.
        :return: The wrapped objective function.
        """
        raise NotImplementedError

    def run(self) -> None:
        """
        Run the optimization process.
        """
        raise NotImplementedError

    @property
    def trials(self):
        """
        :return: The list of trials performed by the tuner.
        """
        return self._trials

    @property
    def best_params(self) -> dict:
        """
        :return: The best hyperparameters found by the tuner.
        """
        return self.best_trial.params

    @property
    def best_result(self) -> Union[float, dict]:
        """
        :return: The best result found by the tuner.
        """
        return self.best_trial.result

    @property
    def results(self) -> list:
        """
        :return: The results of all trials performed by the tuner.
        """
        return [trial.result for trial in self._trials]


class HyperoptTuner(Tuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global hyperopt
        try:
            import hyperopt
        except ImportError:
            raise ImportError("Hyperopt is not installed. Please install hyperopt to use this tuner.")

    def wrap_objective(self, objective: Callable) -> Callable:
        @wraps(objective)
        def wrapped_objective(config: dict) -> dict:
            result_ = objective(config)
            if self.metric is not None:
                if self.metric != "loss":
                    result_['loss'] = result_[self.metric]

                result_['status'] = hyperopt.STATUS_OK

            if self.mode == 'min':
                return result_
            else:  # self.mode == 'max'
                if self.metric is not None:
                    result_['loss'] = -result_['loss']
                else:
                    result_ = -result_
                return result_

        return wrapped_objective

    def run(self) -> None:
        trials = hyperopt.Trials()
        hyperopt_space = self.search_space.to_hyperopt()

        wrapped_hyperopt_objective = self.wrap_objective(self.objective)

        result = hyperopt.fmin(wrapped_hyperopt_objective,
                               hyperopt_space,
                               trials=trials,
                               **self.framework_params)

        best_result = self._parse_result(trials.best_trial['result'])

        self.best_trial = Trial(params=hyperopt.space_eval(hyperopt_space, result),
                                result=best_result)

        for trial in trials.trials:
            self._trials.append(
                Trial(
                    params=hyperopt.space_eval(hyperopt_space, self._parse_misc_vals(trial['misc']['vals'])),
                    result=self._parse_result(trial['result'])
                )
            )

    @staticmethod
    def _parse_misc_vals(vals: dict) -> dict:
        """
        Parse the 'vals' dictionary of hyperopt to return the hyperparameters in the correct format.
        """
        return {key: vals[key][0] for key in vals if vals[key]}

    def _parse_result(self, result: dict) -> Union[float, dict]:
        """
        Parse the result dictionary of hyperopt to return the result defined by the user.
        """
        result = result.copy()
        if self.metric is None:
            result = result['loss']
            if self.mode == 'max':
                result = -result
        else:
            result.pop('status')
            if self.metric != "loss":
                result.pop('loss')

        return result


class OptunaTuner(Tuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global optuna
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is not installed. Please install optuna to use this tuner.")

    def wrap_objective(self, objective: Callable) -> Callable:
        @wraps(objective)
        def wrapped_objective(trial: optuna.Trial):
            config = self.search_space.to_optuna(trial)
            result = objective(config)
            if isinstance(result, dict):
                for key, value in result.items():
                    trial.set_user_attr(key, value)
                return result[self.metric]
            else:
                return result

        return wrapped_objective

    def run(self) -> None:
        mode_optuna = "minimize" if self.mode == "min" else "maximize"
        study_params = dict()
        study_params['sampler'] = self.framework_params.pop('sampler', None)
        study_params['pruner'] = self.framework_params.pop('pruner', None)
        study_params['study_name'] = self.framework_params.pop('study_name', None)

        study = optuna.create_study(direction=mode_optuna, **study_params)

        wrapped_optuna_objective = self.wrap_objective(self.objective)
        study.optimize(wrapped_optuna_objective,
                       **self.framework_params)

        self._params = study.best_trial.params.copy()
        self.best_trial = Trial(params=self._parse_params(study.best_trial.params),
                                result=study.best_trial.value if self.metric is None else study.best_trial.user_attrs)

        for trial in study.trials:
            self._params = trial.params.copy()
            self._trials.append(
                Trial(
                    params=self._parse_params(trial.params),
                    result=trial.value if self.metric is None else trial.user_attrs
                )
            )

        del self._params

    def _parse_params(self, params: dict) -> dict:
        out = dict()
        for k, v in params.items():
            if self.search_space.sep not in k:
                out[k] = v

        return self._parse_params_inner(out, name=self.search_space.name, sep=self.search_space.sep)

    def _parse_params_inner(self, param_cfg: Any,
                            prefix: str = '', name: str = 'name', sep: str = '?') -> Any:
        param_cfg = param_cfg.copy()

        out = dict()

        if isinstance(param_cfg, dict):
            if name in param_cfg.keys():
                name_value = param_cfg.pop(name)
                return {
                    name: name_value,
                    **self._parse_params_inner(param_cfg, add_prefix(prefix, name_value, sep), name, sep)
                }

        for k, v in param_cfg.items():
            if isinstance(v, dict):
                if 'args' in v.keys():
                    out[k] = self._params[add_prefix(prefix, k, sep)]
                else:
                    out[k] = self._parse_params_inner(v, add_prefix(prefix, k, sep), name, sep)
            elif isinstance(v, list):
                selected = self._params[add_prefix(prefix, k, sep)]
                out[k] = self._parse_params_inner(selected, add_prefix(prefix, k, sep), name, sep)
            else:
                out[k] = v

        return out


class FlamlTuner(Tuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global flaml
        try:
            import flaml.tune
        except ImportError:
            raise ImportError("FLAML is not installed. Please install FLAML to use this tuner.")

    def run(self) -> None:
        if self.metric is None:
            result = flaml.tune.run(self.objective,
                                    config=self.search_space.to_flaml(),
                                    mode=self.mode,
                                    **self.framework_params)
        else:
            result = flaml.tune.run(self.objective,
                                    config=self.search_space.to_flaml(),
                                    mode=self.mode,
                                    metric=self.metric,
                                    **self.framework_params)

        best_result = self._parse_result(result.get_best_trial(self.metric, mode=self.mode).last_result)

        self.best_trial = Trial(params=result.best_config,
                                result=best_result)

        for trial in result.trials:
            self._trials.append(
                Trial(
                    params=trial.config,
                    result=self._parse_result(trial.last_result)
                )
            )

    def _parse_result(self, result: dict) -> Union[float, dict]:
        """
        Parse the result trail of flaml to return the result defined by the user.
        """
        result = result.copy()
        if self.metric is None:
            return result['_metric']
        else:
            for key_to_move in ['experiment_tag', 'time_total_s', 'training_iteration']:
                result.pop(key_to_move, None)
            return result


def create_tuner(
        objective: Callable,
        search_space: SearchSpace,
        mode: str = "min",
        metric: Optional[str] = None,
        framework: str = "hyperopt",
        framework_params: dict = None,
        **kwargs
) -> Tuner:
    """
    Create a tuner object based on the specified framework.
    :param objective: user-defined objective function. The objective function can ouput a single float value, or a
    dictionary containing the metric value and other values.
    :param search_space: A SearchSpace object containing the search space for the hyperparameters.
    :param mode: The optimization mode. Either 'min' or 'max'.
    :param metric: If the objective function returns a dictionary, the metric key specifies the key to be used for
    optimization.
    :param framework: The framework to be used for optimization. Supported frameworks are "hyperopt", "optuna", and
    "flaml".
    :param framework_params: Additional parameters to be passed to the framework tuning function.
    :param kwargs: Additional parameters to be passed to the framework tuning function.
    :return: A Tuner object based on the specified framework.
    """
    if framework == "hyperopt":
        return HyperoptTuner(objective=objective,
                             search_space=search_space,
                             mode=mode,
                             metric=metric,
                             framework=framework,
                             framework_params=framework_params,
                             **kwargs)
    elif framework == "optuna":
        return OptunaTuner(objective=objective,
                           search_space=search_space,
                           mode=mode,
                           metric=metric,
                           framework=framework,
                           framework_params=framework_params,
                           **kwargs)
    elif framework == "flaml":
        return FlamlTuner(objective=objective,
                          search_space=search_space,
                          mode=mode,
                          metric=metric,
                          framework=framework,
                          framework_params=framework_params,
                          **kwargs)
    else:
        raise ValueError(f"Framework {framework} not supported")
