import pytest

import hiertunehub


class TestSearchSpaceInit:
    def test_search_space_init_file(self):
        ss = hiertunehub.SearchSpace("./configs/example.yaml")
        assert ss is not None

    def test_search_space_init_dict(self):
        from .configs.example import config
        ss = hiertunehub.SearchSpace.from_dict(config)
        assert ss is not None