import pytest

import hiertunehub


class TestSearchSpaceInit:
    def test_search_space_init_file(self):
        ss = hiertunehub.SearchSpace(config_file="./configs/example.yaml")
        assert ss is not None

    def test_search_space_init_dict(self):
        from .configs.example import config
        ss = hiertunehub.SearchSpace(config=config)
        assert ss is not None