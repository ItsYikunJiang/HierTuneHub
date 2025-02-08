import pytest

import HPSearchSpace


class TestFlaml:
    def test_flaml(self):
        ss = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")
        out = ss.to_flaml()
        assert out is not None
