from flaml import tune

from hiertunehub import SearchSpace

upper = max(5, min(32768, 1000))  # upper must be larger than lower

XGBOOST_SEARCH_SPACE_FLAML = {
    "n_estimators": {
        "domain": tune.lograndint(lower=4, upper=upper),
        "init_value": 4,
        "low_cost_init_value": 4,
    },
    "max_leaves": {
        "domain": tune.lograndint(lower=4, upper=upper),
        "init_value": 4,
        "low_cost_init_value": 4,
    },
    "max_depth": {
        "domain": tune.choice([0, 6, 12]),
        "init_value": 0,
    },
    "min_child_weight": {
        "domain": tune.loguniform(lower=0.001, upper=128),
        "init_value": 1.0,
    },
    "learning_rate": {
        "domain": tune.loguniform(lower=1 / 1024, upper=1.0),
        "init_value": 0.1,
    },
    "subsample": {
        "domain": tune.uniform(lower=0.1, upper=1.0),
        "init_value": 1.0,
    },
    "colsample_bylevel": {
        "domain": tune.uniform(lower=0.01, upper=1.0),
        "init_value": 1.0,
    },
    "colsample_bytree": {
        "domain": tune.uniform(lower=0.01, upper=1.0),
        "init_value": 1.0,
    },
    "reg_alpha": {
        "domain": tune.loguniform(lower=1 / 1024, upper=1024),
        "init_value": 1 / 1024,
    },
    "reg_lambda": {
        "domain": tune.loguniform(lower=1 / 1024, upper=1024),
        "init_value": 1.0,
    },
}


class TestTransformFlaml:
    def test_transform_flaml(self):
        new_config = SearchSpace._transform_flaml(XGBOOST_SEARCH_SPACE_FLAML)
        assert isinstance(new_config, dict)
