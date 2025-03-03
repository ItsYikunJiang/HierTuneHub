import pytest

import hiertunehub


class TestFlaml:
    def test_flaml(self):
        ss = hiertunehub.SearchSpace("./configs/example.yaml")
        out = ss.to_flaml()
        assert out is not None
