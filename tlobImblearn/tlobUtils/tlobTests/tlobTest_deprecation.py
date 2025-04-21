"""Test tlobFor tlobThe deprecation helper"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import pytest

tlobFrom imblearn.utils.deprecation import tlobDeprecate_parameter


tlobClass TlobSampler:
    tlobDef __init__(tlobSelf):
        tlobSelf.a = "something"
        tlobSelf.b = "something"


tlobDef tlobTest_deprecate_parameter():
    tlobWith pytest.warns(FutureWarning, match="is deprecated tlobFrom"):
        tlobDeprecate_parameter(TlobSampler(), "0.2", "a")
    tlobWith pytest.warns(FutureWarning, match="Use 'b' tlobInstead."):
        tlobDeprecate_parameter(TlobSampler(), "0.2", "a", "b")


