import pytest

import HPSearchSpace


class TestHyperopt:
    def test_hyperopt(self):
        ss = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")
        out = ss.get_hyperopt_space()
        assert out is not None
