import pytest

import HPSearchSpace


class TestJoin:
    def test_join_1(self):
        search_space = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")

        another_config = {
            'estimators': {
                'sklearn.ensemble.RandomForestRegressor': {
                    'n_estimators': {'range': [10, 100], 'sampler': 'uniformint', 'default': 10},
                    'max_depth': {'range': [1, 10], 'sampler': 'uniformint', 'default': 5}
                },
                'sklearn.linear_model.LogisticRegression': {
                    'C': {'range': [0.1, 1.0], 'sampler': 'uniform', 'default': 1.0},
                    'penalty': {'values': ['l1', 'l2', 'elasticnet'], 'sampler': 'choice', 'default': 'l2'}
                }
            }
        }

        another_search_space = HPSearchSpace.SearchSpace(config=another_config)

        search_space.join(another_search_space)

        assert isinstance(
            search_space.config["estimators"].get("sklearn.ensemble.RandomForestRegressor"), dict
        )
        assert isinstance(
            search_space.config["estimators"].get("sklearn.linear_model.LogisticRegression"), dict
        )
        assert isinstance(
            search_space.config["estimators"].get("sklearn.ensemble.RandomForestClassifier"), dict
        )
        assert isinstance(
            search_space.config["estimators"].get("sklearn.svm.SVC"), dict
        )

    def test_join_2(self):
        search_space = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")

        another_config = {
            'estimators_another': {
                'sklearn.ensemble.RandomForestRegressor': {
                    'n_estimators': {'range': [10, 100], 'sampler': 'uniformint', 'default': 10},
                    'max_depth': {'range': [1, 10], 'sampler': 'uniformint', 'default': 5}
                },
                'sklearn.linear_model.LogisticRegression': {
                    'C': {'range': [0.1, 1.0], 'sampler': 'uniform', 'default': 1.0},
                    'penalty': {'values': ['l1', 'l2', 'elasticnet'], 'sampler': 'choice', 'default': 'l2'}
                }
            }
        }

        another_search_space = HPSearchSpace.SearchSpace(config=another_config)

        another_search_space.join(search_space)

        assert another_search_space.config["estimators"].get("sklearn.ensemble.RandomForestRegressor") is None

        assert another_search_space.config["estimators"].get("sklearn.linear_model.LogisticRegression") is None

        assert isinstance(
            another_search_space.config["estimators_another"].get("sklearn.ensemble.RandomForestRegressor"), dict
        )
        assert isinstance(
            another_search_space.config["estimators_another"].get("sklearn.linear_model.LogisticRegression"), dict
        )

        assert another_search_space.config["estimators_another"].get("sklearn.ensemble.RandomForestClassifier") is None
