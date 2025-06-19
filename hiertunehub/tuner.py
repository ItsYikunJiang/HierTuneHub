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
                 init_config: Optional[list[dict]] = None,
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
        :param init_config: The initial configuration to be used for the tuning process.
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

        # TODO: Add support for init_config

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

    def resume_from_results(self, results: Union[str, list[dict]]) -> None:
        """
        Resume the optimization process from a saved state.
        :param results: A list containing the results of previous trials or a path to a file containing the saved results.
        """
        raise NotImplementedError

    @staticmethod
    def load_result(result_path: str) -> list[dict]:
        """
        Load the results from a saved state.
        :param result_path: The path to the file containing the saved results.
        """
        import json
        return json.load(open(result_path, 'r'))

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

    def result_to_file(self, name: str, path_to_file: str,
                       uuid: bool = True,
                       dump_short: bool = True,
                       include_result_keys: list = None) -> None:
        """
        Save the results of the tuner to a file.
        :param name: The name of the file.
        :param path_to_file: The path to the file where the results will be saved.
        :param uuid: Whether to add a unique identifier to the file name.
        :param dump_short: Whether to dump only the metric value or the full result dictionary.
        :param include_result_keys: A list of keys to include in the result dictionary.
        """
        import json
        import os

        if not os.path.exists(path_to_file):
            os.makedirs(path_to_file)

        # Specify file name
        if uuid:
            import uuid
            uuid_file = uuid.uuid4()
            full_path = f"{path_to_file}/{name}_{uuid_file}.json"
        else:
            full_path = f"{path_to_file}/{name}.json"

        # Test if file exists. If it does, generate another uuid or raise an error
        while os.path.exists(full_path):
            if uuid:
                uuid_file = uuid.uuid4()
                full_path = f"{path_to_file}/{name}_{uuid_file}.json"
            else:
                raise FileExistsError(f"File {full_path} already exists")

        if not dump_short:
            out = [
                {
                    'params': trial.params,
                    'result': trial.result
                } for trial in self._trials
            ]
        else:
            out = [
                {
                    'params': trial.params,
                    'result': trial.result if self.metric is None else trial.result[self.metric]
                } for trial in self._trials
            ]

        if include_result_keys is not None:
            out = [
                {
                    'params': trial.params,
                    'result': {key: trial.result.get(key, None) for key in include_result_keys}
                } for trial in self._trials
            ]

        with open(full_path, 'w') as file:
            json.dump(out, file, indent=4, default=Tuner._dump_default)

    @staticmethod
    def _dump_default(obj):
        global numpy
        try:
            import numpy
        except ImportError:
            numpy = None
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return "<not serializable>"


class HyperoptTuner(Tuner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        global hyperopt
        try:
            import hyperopt
        except ImportError:
            raise ImportError("Hyperopt is not installed. Please install hyperopt to use this tuner.")

        self.hp_trials = hyperopt.Trials()

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
        hyperopt_space = self.search_space.to_hyperopt()

        wrapped_hyperopt_objective = self.wrap_objective(self.objective)

        hp_result = hyperopt.fmin(wrapped_hyperopt_objective,
                                  hyperopt_space,
                                  trials=self.hp_trials,
                                  **self.framework_params)

        best_result = self._parse_result(self.hp_trials.best_trial['result'])

        self.best_trial = Trial(params=hyperopt.space_eval(hyperopt_space, hp_result),
                                result=best_result)

        for trial in self.hp_trials.trials:
            self._trials.append(
                Trial(
                    params=hyperopt.space_eval(hyperopt_space, self._parse_misc_vals(trial['misc']['vals'])),
                    result=self._parse_result(trial['result'])
                )
            )

    def resume_from_results(self, results: Union[str, list[dict]]) -> None:
        past_results = self.load_result(results) if isinstance(results, str) else results
        if not past_results:
            raise ValueError("No results to resume from")

        points = [trial['params'] for trial in past_results]
        results = [self.wrap_objective(lambda config: trial['result'])(trial['params']) for trial in past_results]

        self.hp_trials = self._generate_trials(points, results)
        self.hp_trials.refresh()

        self.run()

    def _generate_trials(
            self,
            points: list,
            results: Optional[list] = None
    ) -> "hyperopt.Trials":
        """
        Generate a new Trials object with a list of points to evaluate.
        If results are provided, they will be added to the Trials object and no evaluation will be performed.
        """

        def _generate_trial(tid_: int,
                            idxs_: dict,
                            vals_: dict,
                            result: Optional[dict] = None
                            ) -> dict:
            """
            Generate a single trial dictionary with the given trial id and search space.
            """
            doc = {
                "state": 0,  # hyperopt.JOB_STATE_NEW
                "tid": tid_,
                "spec": None,
                "result": {"status": "new"},
                "misc": {
                    "tid": tid_,
                    "cmd": ("domain_attachment", "FMinIter_Domain"),
                    "workdir": None,
                    "idxs": idxs_,
                    "vals": vals_,
                },
                "exp_key": None,
                "owner": None,
                "version": 0,
                "book_time": None,
                "refresh_time": None,
            }

            if result is not None:
                doc['result'] = result
                doc['state'] = 2  # hyperopt.JOB_STATE_DONE

            return doc

        trials = hyperopt.Trials()
        domain_temp = hyperopt.Domain(lambda x: None, self.search_space.to_hyperopt())
        keys = domain_temp.params.keys()
        new_docs = []
        for i, point in enumerate(points):
            point_hp = self.search_space.point_to_hyperopt_representation(point)
            idxs = {key: [] for key in keys}
            vals = {key: [] for key in keys}
            for k, v in point_hp.items():
                idxs[k] = [i]
                vals[k] = [v]
            new_doc = _generate_trial(
                i,
                idxs,
                vals,
                results[i]
            )
            new_docs.append(new_doc)
        trials.insert_trial_docs(new_docs)
        return trials

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

        global optuna, op_dis
        try:
            import optuna
            import optuna.distributions as op_dis
        except ImportError:
            raise ImportError("Optuna is not installed. Please install optuna to use this tuner.")

        self.study = None

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

    def _create_or_load_study(self) -> 'optuna.Study':
        if self.study is not None:
            return self.study

        mode_optuna = "minimize" if self.mode == "min" else "maximize"
        study_params = dict()
        study_params['sampler'] = self.framework_params.pop('sampler', None)
        study_params['pruner'] = self.framework_params.pop('pruner', None)
        study_params['study_name'] = self.framework_params.pop('study_name', None)

        study = optuna.create_study(direction=mode_optuna, **study_params)
        return study

    def run(self) -> None:
        self.study = self._create_or_load_study()

        wrapped_optuna_objective = self.wrap_objective(self.objective)
        self.study.optimize(wrapped_optuna_objective,
                            **self.framework_params)

        self._params = self.study.best_trial.params.copy()
        self.best_trial = Trial(params=self._parse_params(self.study.best_trial.params),
                                result=self.study.best_trial.value
                                if self.metric is None else self.study.best_trial.user_attrs)

        for trial in self.study.trials:
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

    def resume_from_results(self, results: Union[str, list[dict]]) -> None:
        past_results = self.load_result(results) if isinstance(results, str) else results
        self.study = self._create_or_load_study()

        for i, trial in enumerate(past_results):
            params = trial['params']
            result = trial['result']
            if self.metric is not None:
                value = result[self.metric] if self.metric in result else result
                user_attrs = result
            else:
                value = result
                user_attrs = None

            op_params, op_distribution = self.search_space.point_to_optuna_representation(params)

            # Create a new Optuna trial
            optuna_trial = optuna.create_trial(
                params=op_params,
                distributions=op_distribution,
                value=value,
                user_attrs=user_attrs,
            )

            self.study.add_trial(optuna_trial)

        if 'n_trials' in self.framework_params:
            self.framework_params['n_trials'] -= len(past_results)

        self.run()

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

    def resume_from_results(self, results: Union[str, list[dict]]) -> None:
        past_results = self.load_result(results) if isinstance(results, str) else results

        points_evaluated = []
        rewards = []
        for trial in past_results:
            points_evaluated.append(trial['params'])
            result = trial['result']
            if isinstance(result, dict):
                rewards.append(result[self.metric])
            else:
                rewards.append(result)

            self._trials.append(
                Trial(
                    params=trial['params'],
                    result=result
                )
            )

        self.framework_params['points_to_evaluate'] = points_evaluated
        self.framework_params['evaluated_rewards'] = rewards
        if 'num_samples' in self.framework_params:
            self.framework_params['num_samples'] -= len(past_results)

        self.run()


def create_tuner(
        objective: Callable,
        search_space: SearchSpace,
        mode: str = "min",
        metric: Optional[str] = None,
        init_config: Optional[list[dict]] = None,
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
    :param init_config: The initial configuration to be used for the tuning process.
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
