import random

import pytest
import time

from hiertunehub import SearchSpace, create_tuner

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

iris = load_iris()
x, y = iris.data, iris.target


class TestTuner:
    search_space = SearchSpace(config_file="./configs/example.yaml")

    @staticmethod
    def objective(config):
        # estimator = config['estimators']['estimator_class']
        # params = config['estimators']['params']
        # model = estimator(**params)
        score = 0
        return random.random()

    @staticmethod
    def objective_complex(config):
        t1 = time.time()
        # estimator = config['estimators']['estimator_class']
        # params = config['estimators']['params']
        # model = estimator(**params)
        score = random.random()
        t2 = time.time()
        return {
            "score": score,
            "time": t2 - t1
        }

    def test_tuner_hyperopt(self):
        tuner = create_tuner(objective=self.objective,
                             search_space=self.search_space,
                             mode="max",
                             framework="hyperopt",
                             framework_params={"max_evals": 10})
        tuner.run()
        assert isinstance(tuner.best_params, dict)
        assert isinstance(tuner.best_result, float)

    def test_tuner_optuna(self):
        tuner = create_tuner(objective=self.objective,
                             search_space=self.search_space,
                             framework="optuna",
                             framework_params={"n_trials": 10})
        tuner.run()
        assert isinstance(tuner.best_params, dict)
        assert isinstance(tuner.best_result, float)

    def test_tuner_flaml(self):
        tuner = create_tuner(objective=self.objective,
                             search_space=self.search_space,
                             framework="flaml",
                             framework_params={"num_samples": 10})
        tuner.run()
        assert isinstance(tuner.best_params, dict)
        assert isinstance(tuner.best_result, float)

    def test_tuner_hyperopt_complex(self):
        tuner = create_tuner(objective=self.objective_complex,
                             search_space=self.search_space,
                             mode="max",
                             metric="score",
                             framework="hyperopt",
                             framework_params={"max_evals": 10})
        tuner.run()
        assert isinstance(tuner.best_params, dict)
        assert list(tuner.best_result.keys()) == ["score", "time"]

    def test_tuner_optuna_complex(self):
        tuner = create_tuner(objective=self.objective_complex,
                             search_space=self.search_space,
                             metric="score",
                             framework="optuna",
                             framework_params={"n_trials": 10})
        tuner.run()
        assert isinstance(tuner.best_params, dict)
        assert list(tuner.best_result.keys()) == ["score", "time"]

    def test_tuner_flaml_complex(self):
        tuner = create_tuner(objective=self.objective_complex,
                             search_space=self.search_space,
                             metric="score",
                             framework="flaml",
                             framework_params={"num_samples": 10})
        tuner.run()
        assert isinstance(tuner.best_params, dict)
        assert list(tuner.best_result.keys()) == ["score", "time"]
