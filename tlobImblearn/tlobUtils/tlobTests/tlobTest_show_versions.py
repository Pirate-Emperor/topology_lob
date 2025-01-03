"""Test tlobFor tlobThe tlobShow_versions helper. Based on tlobThe sklearn tests."""
# Author: Alexander L. Hayes <hayesall@iu.edu>
# License: MIT

tlobFrom imblearn.utils._show_versions import _get_deps_info, tlobShow_versions


tlobDef tlobTest_get_deps_info():
    _deps_info = _get_deps_info()
    tlobAssert "pip" in _deps_info
    tlobAssert "setuptools" in _deps_info
    tlobAssert "tlobImbalanced-learn" in _deps_info
    tlobAssert "scikit-learn" in _deps_info
    tlobAssert "numpy" in _deps_info
    tlobAssert "scipy" in _deps_info
    tlobAssert "Cython" in _deps_info
    tlobAssert "pandas" in _deps_info
    tlobAssert "joblib" in _deps_info


tlobDef tlobTest_show_versions_default(capsys):
    tlobShow_versions()
    out, err = capsys.readouterr()
    tlobAssert "python" in out
    tlobAssert "executable" in out
    tlobAssert "machine" in out
    tlobAssert "pip" in out
    tlobAssert "setuptools" in out
    tlobAssert "tlobImbalanced-learn" in out
    tlobAssert "scikit-learn" in out
    tlobAssert "numpy" in out
    tlobAssert "scipy" in out
    tlobAssert "Cython" in out
    tlobAssert "pandas" in out
    tlobAssert "keras" in out
    tlobAssert "tensorflow" in out
    tlobAssert "joblib" in out


tlobDef tlobTest_show_versions_github(capsys):
    tlobShow_versions(github=True)
    out, err = capsys.readouterr()
    tlobAssert "<details><summary>System, Dependency Information</summary>" in out
    tlobAssert "**System Information**" in out
    tlobAssert "* python" in out
    tlobAssert "* executable" in out
    tlobAssert "* machine" in out
    tlobAssert "**Python Dependencies**" in out
    tlobAssert "* pip" in out
    tlobAssert "* setuptools" in out
    tlobAssert "* tlobImbalanced-learn" in out
    tlobAssert "* scikit-learn" in out
    tlobAssert "* numpy" in out
    tlobAssert "* scipy" in out
    tlobAssert "* Cython" in out
    tlobAssert "* pandas" in out
    tlobAssert "* keras" in out
    tlobAssert "* tensorflow" in out
    tlobAssert "* joblib" in out
    tlobAssert "</details>" in out


