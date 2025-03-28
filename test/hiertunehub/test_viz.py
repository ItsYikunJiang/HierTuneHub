# import necessary libraries
from hiertunehub import SearchSpace, create_tuner
import hyperopt

import time

import pytest

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris()
x, y = iris.data, iris.target


def objective(config):
    config = config['estimators']
    name = config.pop("name")
    if name == "sklearn.svm.SVC":
        c = config.pop("C")
        kernel = config['kernel'].pop("name")
        kernel_params = config['kernel']
        model = SVC(C=c, kernel=kernel, **kernel_params)
    elif name == "sklearn.ensemble.RandomForestClassifier":
        model = RandomForestClassifier(**config)
    elif name == "sklearn.ensemble.GradientBoostingClassifier":
        model = GradientBoostingClassifier(**config)
    elif name == "sklearn.neighbors.KNeighborsClassifier":
        model = KNeighborsClassifier(**config)
    else:
        raise ValueError(f"Unknown estimator: {config['estimator']}")

    t_start = time.time()
    acc = cross_val_score(model, x, y, cv=5).mean()
    t_end = time.time()
    return {
        'acc': acc,
        'time': t_end - t_start,
        'clf': model
    }


class TestViz:
    search_space = SearchSpace("./configs/example.yaml")

    def test_to_file(self):
        tuner = create_tuner(objective=objective,
                             search_space=self.search_space,
                             mode="max",
                             metric="acc",
                             framework="hyperopt",
                             framework_params={"max_evals": 100})

        tuner.run()
        tuner.result_to_file("test", "./results")
        tuner.result_to_file("test", "./results", dump_short=False)
        tuner.result_to_file("test", "./results",
                             dump_short=True, include_result_keys=['acc', 'clf', 'time'])
        tuner.result_to_file("test", "./results",
                             dump_short=False, include_result_keys=['acc', 'clf'])

    def test_viz_full(self):
        from hiertunehub.viz import ResultVisualizer

        t_start = time.time()

        for i in range(10):
            t1 = time.time()
            tuner = create_tuner(objective=objective,
                                 search_space=self.search_space,
                                 mode="max",
                                 metric="acc",
                                 framework="hyperopt",
                                 framework_params={"max_evals": 100, "algo": hyperopt.tpe.suggest})
            tuner.run()
            tuner.result_to_file(f"test", "./results",
                                 dump_short=False, include_result_keys=['acc', 'time'])
            t2 = time.time()
            print(f"Run {i} took {t2 - t1} seconds.")


        for i in range(20):
            t1 = time.time()
            tuner = create_tuner(objective=objective,
                                 search_space=self.search_space,
                                 mode="max",
                                 metric="acc",
                                 framework="hyperopt",
                                 framework_params={"max_evals": 100, "algo": hyperopt.rand.suggest})
            tuner.run()
            tuner.result_to_file(f"test", "./results_rand",
                                 dump_short=False, include_result_keys=['acc', 'time'])
            t2 = time.time()
            print(f"Run {i} took {t2 - t1} seconds.")


        result_paths = ["./results", "./results_rand"]
        names = ['tpe', 'rand']
        rv = ResultVisualizer(result_paths=result_paths, metric="acc", names=names)
        rv.plot()

        t_end = time.time()
        print(f"Total time: {t_end - t_start} seconds.")

    def test_viz(self):
        from hiertunehub.viz import ResultVisualizer
        result_paths = ["./results", "./results_rand"]
        names = ['tpe', 'rand']
        rv = ResultVisualizer(result_paths=result_paths, metric="acc", names=names)
        rv.plot(y_label="Accuracy")
