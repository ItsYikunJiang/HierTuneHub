import pytest

import HPSearchSpace


class TestFlaml:
    def test_flaml(self):
        ss = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")
        out = ss.get_flaml_space()
        assert out is not None
