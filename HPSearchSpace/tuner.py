from typing import Callable, Optional, Union
from functools import wraps

from .search_space import SearchSpace

import hyperopt
import optuna
import flaml.tune


class Tuner:
    def __init__(self,
                 objective: Callable,
                 search_space: SearchSpace,
                 mode: str = "min",
                 metric: Optional[str] = None,
                 framework: str = "hyperopt",
                 framework_params: dict = None,
                 ):
        self.objective = objective
        self.search_space = search_space

        self.mode = mode
        if self.mode not in ["min", "max"]:
            raise ValueError("Mode must be either 'min' or 'max'")

        self.framework = framework
        if framework_params is None:
            framework_params = {}
        self.framework_params = framework_params

        self.metric = metric

        self.best_trial = None

    def run(self) -> None:
        match self.framework:
            case "hyperopt":
                self._run_hyperopt()
            case "optuna":
                self._run_optuna()
            case "flaml":
                self._run_flaml()
            case _:
                raise ValueError(f"Framework {self.framework} is not supported")

    @property
    def best_params(self) -> dict:
        return self.best_trial['params']

    @property
    def best_result(self) -> Union[float, dict]:
        return self.best_trial['result']

    def wrap_hyperopt_objective(self, objective: Callable) -> Callable:
        @wraps(objective)
        def wrapped_objective(config: dict) -> dict:
            result_ = objective(config)
            if self.metric is not None:
                if self.metric != "loss":
                    result_['loss'] = result_[self.metric]

                result_['status'] = hyperopt.STATUS_OK

            if self.mode == 'min':
                return result_
            else:  # mode == 'max'
                if self.metric is not None:
                    result_['loss'] = -result_['loss']
                else:
                    result_ = -result_
                return result_

        return wrapped_objective

    def _run_hyperopt(self) -> None:
        trials = hyperopt.Trials()
        hyperopt_space = self.search_space.to_hyperopt()
        if self.metric is None:
            wrapped_hyperopt_objective = self.wrap_hyperopt_objective(self.objective)
            result = hyperopt.fmin(wrapped_hyperopt_objective,
                                   hyperopt_space,
                                   trials=trials,
                                   **self.framework_params)
            best_result = trials.best_trial['result']['loss']
            if self.mode == 'max':
                best_result = -best_result
        else:
            wrapped_hyperopt_objective = self.wrap_hyperopt_objective(self.objective)

            result = hyperopt.fmin(wrapped_hyperopt_objective,
                                   hyperopt_space,
                                   trials=trials,
                                   **self.framework_params)
            best_result = trials.best_trial['result']
            if self.metric != "loss":
                best_result.pop('loss')
            best_result.pop('status')

            if self.mode == 'max':
                best_result[self.metric] = -best_result[self.metric]

        self.best_trial = {
            'params': hyperopt.space_eval(hyperopt_space, result),
            'result': best_result
        }

    def _run_optuna(self) -> None:
        mode_optuna = "minimize" if self.mode == "min" else "maximize"
        study = optuna.create_study(direction=mode_optuna)

        def wrap_optuna_objective(objective: Callable):
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

        wrapped_optuna_objective = wrap_optuna_objective(self.objective)
        study.optimize(wrapped_optuna_objective,
                       **self.framework_params)

        if self.metric is None:
            self.best_trial = {
                'params': study.best_trial.params,
                'result': study.best_trial.value,
            }
        else:
            self.best_trial = {
                'params': study.best_params,
                'result': study.best_trial.user_attrs,
            }

    def _run_flaml(self) -> None:
        if self.metric is None:
            result = flaml.tune.run(self.objective,
                                    config=self.search_space.to_flaml(),
                                    mode=self.mode,
                                    **self.framework_params)
            best_result = result.get_best_trial(self.metric, mode=self.mode).last_result['_metric']
        else:
            result = flaml.tune.run(self.objective,
                                    config=self.search_space.to_flaml(),
                                    mode=self.mode,
                                    metric=self.metric,
                                    **self.framework_params)
            best_result = result.get_best_trial(self.metric, mode=self.mode).last_result
            for key_to_move in ['config', 'config/estimators', 'experiment_tag', 'time_total_s', 'training_iteration']:
                best_result.pop(key_to_move, None)

        self.best_trial = {
            'params': result.best_config,
            'result': best_result
        }
