import pytest

import hiertunehub


class TestHyperopt:
    def test_hyperopt(self):
        ss = hiertunehub.SearchSpace(config_file="./configs/example.yaml")
        out = ss.to_hyperopt()
        assert out is not None
