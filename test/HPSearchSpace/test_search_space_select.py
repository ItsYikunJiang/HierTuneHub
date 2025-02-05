import pytest

import HPSearchSpace


class TestSelect:
    def test_select(self):
        search_space = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")

        selection = {
            "estimators": ["sklearn.svm.SVC", "sklearn.ensemble.RandomForestClassifier"]
        }

        search_space.select(selection)

        assert isinstance(
            search_space.config["estimators"].get("sklearn.ensemble.RandomForestClassifier"), dict
        )
        assert search_space.config["estimators"].get("sklearn.ensemble.GradientBoostingClassifier") is None
        assert search_space.config["estimators"].get("sklearn.neighbors.KNeighborsClassifier") is None
