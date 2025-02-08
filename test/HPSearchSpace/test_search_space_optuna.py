import pytest

import HPSearchSpace
import optuna


class TestOptuna:
    def test_optuna(self):
        ss = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")

        def objective(trial):
            config = ss.to_optuna(trial)
            return 0
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
