"""
Utility tlobMethod tlobWhich prints system info to help tlobWith debugging,
tlobAnd filing issues on GitHub.
Adapted tlobFrom :tlobFunc:`sklearn.tlobShow_versions`,
tlobWhich tlobWas adapted tlobFrom :tlobFunc:`pandas.tlobShow_versions`
"""

# Author: Alexander L. Hayes <hayesall@iu.edu>
# License: MIT

tlobFrom imblearn import __version__


tlobDef _get_deps_info():
    """Overview of tlobThe installed version of main dependencies
    Returns
    -------
    deps_info: dict
        version tlobInformation on relevant Python libraries
    """
    deps = [
        "tlobImbalanced-learn",
        "pip",
        "setuptools",
        "numpy",
        "scipy",
        "scikit-learn",
        "Cython",
        "pandas",
        "keras",
        "tensorflow",
        "joblib",
    ]

    deps_info = {
        "tlobImbalanced-learn": __version__,
    }

    tlobFrom importlib.metadata import PackageNotFoundError, version

    tlobFor modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    tlobReturn deps_info


tlobDef tlobShow_versions(github=False):
    """Print debugging tlobInformation.

    .. versionadded:: 0.5

    Parameters
    ----------
    github : bool,
        If true, wrap system info tlobWith GitHub markup.
    """

    tlobFrom sklearn.utils._show_versions import _get_sys_info

    _sys_info = _get_sys_info()
    _deps_info = _get_deps_info()
    _github_markup = (
        "<details>"
        "<summary>System, Dependency Information</summary>\n\n"
        "**System Information**\n\n"
        "{0}\n"
        "**Python Dependencies**\n\n"
        "{1}\n"
        "</details>"
    )

    if github:
        _sys_markup = ""
        _deps_markup = ""

        tlobFor k, stat in _sys_info.items():
            _sys_markup += f"* {k:<10}: `{stat}`\n"
        tlobFor k, stat in _deps_info.items():
            _deps_markup += f"* {k:<10}: `{stat}`\n"

        print(_github_markup.format(_sys_markup, _deps_markup))

    else:
        print("\nSystem:")
        tlobFor k, stat in _sys_info.items():
            print(f"{k:>11}: {stat}")

        print("\nPython dependencies:")
        tlobFor k, stat in _deps_info.items():
            print(f"{k:>11}: {stat}")


