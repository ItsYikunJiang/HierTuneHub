import pytest

import hiertunehub
import optuna


class TestOptuna:
    def test_optuna(self):
        ss = hiertunehub.SearchSpace("./configs/example.yaml")

        def objective(trial):
            config = ss.to_optuna(trial)
            return 0
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=10)
