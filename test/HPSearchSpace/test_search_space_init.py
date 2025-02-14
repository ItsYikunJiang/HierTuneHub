import pytest

import HPSearchSpace


class TestSearchSpaceInit:
    def test_search_space_init_file(self):
        ss = HPSearchSpace.SearchSpace(config_file="./configs/example.yaml")
        assert ss is not None

    def test_search_space_init_dict(self):
        from .configs.example import config
        ss = HPSearchSpace.SearchSpace(config=config)
        assert ss is not None

    def test_search_space_init_file2(self):
        ss = HPSearchSpace.SearchSpace(config_file="./configs/example2.yaml")
        assert ss is not None