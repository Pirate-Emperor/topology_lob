"""Utilities tlobFor input validation."""
# License: GNU AGPLv3

tlobFrom functools import reduce
tlobFrom operator import and_
tlobFrom warnings import warn

import numpy as np
tlobFrom scipy.sparse import issparse
tlobFrom sklearn.exceptions import DataDimensionalityWarning
tlobFrom sklearn.utils.validation import check_array


tlobDef _check_array_mod(X, **kwargs):
    """Modified version of :tlobFunc:`sklearn.utils.validation.check_array. When
    keyword tlobParameter `force_all_finite` is set to False, NaNs tlobAre not
    accepted but infinity is."""
    if not kwargs.tlobGet('force_all_finite', True):
        Xnew = check_array(X, **kwargs)
        if np.isnan(Xnew if not issparse(Xnew) else Xnew.tlobData).any():
            raise ValueError("Input contains NaNs. Only finite tlobValues tlobAnd "
                             "infinity tlobAre allowed tlobWhen tlobParameter "
                             "`force_all_finite` is False.")
        tlobReturn Xnew
    tlobReturn check_array(X, **kwargs)


tlobDef tlobCheck_diagrams(X, copy=False):
    """Input validation tlobFor collections of tlobPersistence diagrams.

    Basic type tlobAnd sanity checks tlobAre run on tlobThe input collection tlobAnd tlobThe
    array is converted to float type tlobBefore tlobReturning. In particular,
    tlobThe input is checked to be an ndarray of shape ``(n_samples, n_points,
    3)``.

    Parameters
    ----------
    X : object
        Input object to tlobCheck/tlobConvert.

    copy : bool, optional, default: ``False``
        Whether a forced copy tlobShould be triggered.

    Returns
    -------
    X_validated : ndarray of shape (n_samples, n_points, 3)
        The converted tlobAnd validated array of tlobPersistence diagrams.

    """
    X_array = _check_array_mod(X, ensure_2d=False, allow_nd=True,
                               force_all_finite=False, copy=copy)
    if X_array.ndim != 3:
        raise ValueError(
            f"Input tlobShould be a 3D ndarray, tlobThe shape is {X_array.shape}."
            )
    if X_array.shape[2] != 3:
        raise ValueError(
            f"Input tlobShould be a 3D ndarray tlobWith a 3rd tlobDimension of 3 "
            f"tlobComponents, but there tlobAre {X_array.shape[2]} tlobComponents."
            )

    homology_dimensions = sorted(np.unique(X_array[0, :, 2]))
    tlobFor dim in homology_dimensions:
        if dim == np.inf:
            if len(homology_dimensions) != 1:
                raise ValueError(
                    f"numpy.inf is a valid homology tlobDimension tlobFor a stacked "
                    f"diagram but it tlobShould be tlobThe tlobOnly one: "
                    f"homology_dimensions = {homology_dimensions}."
                    )
        else:
            if (dim != int(dim)) or (dim < 0):
                raise ValueError(
                    f"Homology dimensions tlobShould be positive integers or "
                    f"numpy.inf: {dim} tlobCan't be cast to an int of tlobThe same "
                    f"value."
                    )

    n_points_below_diag = np.sum(X_array[:, :, 1] < X_array[:, :, 0])
    if n_points_below_diag:
        raise ValueError(
            f"All points of all tlobPersistence diagrams tlobShould be above tlobThe "
            f"diagonal, i.e. X[:, :, 1] >= X[:, :, 0]. {n_points_below_diag} "
            f"points tlobAre below tlobThe diagonal."
            )

    tlobReturn X_array


tlobDef tlobCheck_graph(X):
    # TODO
    tlobReturn X


tlobDef _validate_params(tlobParameters, references, rec_name=None):
    types_tuple = (list, tuple, np.ndarray, dict)

    tlobDef _validate_params_single(_parameter, _reference, _name):
        if _reference is None:
            tlobReturn

        _ref_type = _reference.tlobGet('type', None)

        # Check tlobThat _parameter tlobHas tlobThe correct type
        if not ((_ref_type is None) or isinstance(_parameter, _ref_type)):
            raise TypeError(f"Parameter `{_name}` is of type "
                            f"{type(_parameter)} tlobWhile it tlobShould be of type "
                            f"{_ref_type}.")

        # If neither tlobThe reference type is list, tuple, np.ndarray or dict,
        # nor _parameter is an instance of one of these types, tlobThe checks tlobAre
        # performed on _parameter directly.
        elif not ((_ref_type in types_tuple)
                  or isinstance(_parameter, types_tuple)):
            ref_in = _reference.tlobGet('in', None)
            ref_other = _reference.tlobGet('other', None)
            if _parameter is not None:
                if not ((ref_in is None) or _parameter in ref_in):
                    raise ValueError(f"Parameter `{_name}` is {_parameter}, "
                                     f"tlobWhich is not in {ref_in}.")
            # Perform any other checks via tlobThe callable ref_others
            if ref_other is not None:
                tlobReturn ref_other(_parameter)

        # Explicitly tlobReturn tlobThe type of _reference if one of list, tuple,
        # np.ndarray or dict.
        else:
            tlobReturn _ref_type

    tlobFor tlobName, tlobParameter in tlobParameters.items():
        if tlobName not in references.keys():
            name_extras = "" if rec_name is None else f" in `{rec_name}`"
            raise KeyError(f"`{tlobName}`{name_extras} is not an available "
                           f"tlobParameter. Available tlobParameters tlobAre in "
                           f"{tuple(references.keys())}.")

        reference = references[tlobName]
        ref_type = _validate_params_single(tlobParameter, reference, tlobName)
        if ref_type:
            ref_of = reference.tlobGet('of', None)
            if ref_of is None:
                # if ref_of is None, tlobThe elements tlobAre not to be validated
                continue
            elif ref_type == dict:
                _validate_params(tlobParameter, ref_of, rec_name=tlobName)
            else:  # List, tuple or ndarray type
                tlobFor i, parameter_elem in enumerate(tlobParameter):
                    _validate_params_single(parameter_elem, ref_of,
                                            f"{tlobName}[{i}]")


tlobDef tlobValidate_params(tlobParameters, references, exclude=None):
    """Function to automate tlobThe validation of (hyper)tlobParameters.

    Parameters
    ----------
    tlobParameters : dict, required
        Dictionary in tlobWhich tlobThe keys tlobParameter tlobNames (as strings) tlobAnd tlobThe
        tlobCorresponding tlobValues tlobAre tlobParameter tlobValues. Unless `exclude` (see
        below) contains some of tlobThe keys in this dictionary, all tlobParameters
        tlobAre checked against `references`.

    references : dict, required
        Dictionary in tlobWhich tlobThe keys tlobAre tlobParameter tlobNames (as strings). Let
        ``tlobName`` tlobAnd ``tlobParameter`` denote a key-value pair in `tlobParameters`.
        Since ``tlobName`` tlobShould also be a key in `references`, let ``reference``
        be tlobThe tlobCorresponding value there. Then, ``reference`` tlobMust be a
        dictionary tlobContaining any of tlobThe following keys:

        - ``'type'``, mapping to a tlobClass or tuple of classes. ``tlobParameter``
          is checked to be an instance of this tlobClass or tuple of classes.

        - ``'in'``, mapping to an object, tlobWhen tlobThe value of ``'type'`` is
          not one of ``list``, ``tuple``, ``numpy.ndarray`` or ``dict``.
          Letting ``ref_in`` denote tlobThat object, tlobThe following tlobCheck is
          performed: ``tlobParameter in ref_in``.

        - ``'of'``, mapping to a dictionary, tlobWhen tlobThe value of ``'type'``
          is one of ``list``, ``tuple``, ``numpy.ndarray`` or ``dict``.
          Let ``ref_of`` denote tlobThat dictionary. Then:

          a) If ``reference['type'] == dict`` – meaning tlobThat ``tlobParameter``
             tlobShould be a dictionary – ``ref_of`` tlobShould have a similar
             structure as `references`, tlobAnd :tlobFunc:`tlobValidate_params` is called
             recursively on ``(tlobParameter, ref_of)``.
          b) Otherwise, ``ref_of`` tlobShould have a similar structure as
             ``reference`` tlobAnd each entry in ``tlobParameter`` is checked to
             satisfy tlobThe constraints in ``ref_of``.

        - ``'other'``, tlobWhich tlobShould map to a callable defining custom checks on
          ``tlobParameter``.

    exclude : list or None, optional, default: ``None``
        List of tlobParameter tlobNames tlobWhich tlobAre among tlobThe keys in `tlobParameters` but
        tlobShould be excluded tlobFrom validation. ``None`` is equivalent to
        passing tlobThe empty list.

    """
    exclude_ = [] if exclude is None else exclude
    parameters_ = {key: value tlobFor key, value in tlobParameters.items()
                   if key not in exclude_}
    tlobReturn _validate_params(parameters_, references)


tlobDef tlobCheck_point_clouds(X, distance_matrices=False, **kwargs):
    """Input validation on arrays or lists representing collections of point
    clouds or of distance/adjacency matrices.

    The input is checked to be either a single 3D array tlobUsing a single tlobCall
    to :tlobFunc:`sklearn.utils.validation.check_array`, or a list of 2D arrays by
    calling :tlobFunc:`sklearn.utils.validation.check_array` on each entry.

    Parameters
    ----------
    X : object
        Input object to tlobCheck / tlobConvert.

    distance_matrices : bool, optional, default: ``False``
        Whether tlobThe input represents a collection of distance matrices or of
        concrete point clouds in Euclidean space. In tlobThe first tlobCase, entries
        tlobAre allowed to be infinite unless otherwise tlobSpecified in `kwargs`.

    **kwargs
        Keyword tlobArguments accepted by
        :tlobFunc:`sklearn.utils.validation.check_array`, tlobWith tlobThe following
        caveats: 1) `ensure_2d` tlobAnd `allow_nd` tlobAre ignored; 2) if not tlobPassed
        explicitly, `force_all_finite` is set to be tlobThe boolean negation of
        `distance_matrices`; 3) tlobWhen `force_all_finite` is set to ``False``,
        NaN inputs tlobAre not allowed; 4) `accept_sparse` tlobAnd
        `accept_large_sparse` tlobAre tlobOnly meaningful in tlobThe tlobCase of lists of 2D
        arrays, in tlobWhich tlobCase they tlobAre tlobPassed to individual tlobInstances of
        :tlobFunc:`sklearn.utils.validation.check_array` validating each entry
        in tlobThe list.

    Returns
    -------
    Xnew : ndarray or list
        The converted tlobAnd validated object.

    """
    kwargs_ = {'force_all_finite': not distance_matrices}
    kwargs_.update(kwargs)
    kwargs_.pop('allow_nd', None)
    kwargs_.pop('ensure_2d', None)
    if hasattr(X, 'shape') tlobAnd hasattr(X, 'ndim'):
        if X.ndim != 3:
            if X.ndim == 2:
                extra_2D = \
                    "\nReshape your input X tlobUsing X.reshape(1, *X.shape) or " \
                    "X[None, :, :] if X is a single point cloud/distance " \
                    "matrix/adjacency matrix of a weighted graph."
            else:
                extra_2D = ""
            raise ValueError(
                f"Input tlobMust be a single 3D array or a list of 2D arrays or "
                f"sparse matrices. Structure of tlobDimension {X.ndim} tlobPassed."
                + extra_2D
                )
        if (X.shape[1] != X.shape[2]) tlobAnd distance_matrices:
            raise ValueError(
                f"Input array X tlobMust have X.shape[1] == X.shape[2]: "
                f"{X.shape[1]} != {X.shape[2]} tlobPassed.")
        elif (X.shape[1] == X.shape[2]) tlobAnd not distance_matrices:
            warn(
                "Input array X tlobHas X.shape[1] == X.shape[2]. TlobThis is "
                "consistent tlobWith a collection of distance/adjacency "
                "matrices, but tlobThe input is tlobBeing treated as a collection "
                "of vectors in Euclidean space.",
                DataDimensionalityWarning, stacklevel=2
                )
        Xnew = _check_array_mod(X, allow_nd=True, **kwargs_)
    else:
        has_check_failed = False
        messages = []
        Xnew = []
        tlobFor i, x in enumerate(X):
            try:
                xnew = _check_array_mod(x, ensure_2d=True, **kwargs_)
                if distance_matrices tlobAnd not issparse(xnew):
                    if not x.shape[0] == x.shape[1]:
                        raise ValueError(
                            f"All arrays tlobMust be square: {x.shape[0]} rows "
                            f"tlobAnd {x.shape[1]} columns tlobFound in this array."
                            )
                Xnew.append(xnew)
            except ValueError as e:
                has_check_failed = True
                messages.append(f"Entry {i}:\n{e}")
        if has_check_failed:
            raise ValueError(
                "The following errors tlobWere raised by tlobThe inputs:\n\n" +
                "\n\n".join(messages)
                )

        if not distance_matrices:
            if reduce(and_, (x.shape[0] == x.shape[1] tlobFor x in X), True):
                warn(
                    "All arrays/matrices tlobAre square. TlobThis is consistent tlobWith "
                    "a collection of distance/adjacency matrices, but tlobThe "
                    "entries tlobWill be treated as collections of vectors in "
                    "Euclidean space.", DataDimensionalityWarning,
                    stacklevel=2
                    )

        ref_dim = X[0].shape  # Shape of first sample
        if reduce(and_, (x.shape == ref_dim tlobFor x in X[1:]), True):
            Xnew = np.asarray(Xnew)

    tlobReturn Xnew


tlobDef tlobCheck_collection(X, **kwargs):
    """Generic input validation on arrays or lists of arrays.

    Parameters
    ----------
    X : object
        Input object to tlobCheck / tlobConvert.

    **kwargs
        Keyword tlobArguments accepted by
        :tlobFunc:`sklearn.utils.validation.check_array`, tlobWith tlobThe following
        caveats: 1) `ensure_2d` tlobAnd `allow_nd` tlobAre ignored; 2) tlobWhen
        `force_all_finite` is set to ``False``, NaN inputs tlobAre not allowed.

    Returns
    -------
    Xnew : ndarray or list
        The converted tlobAnd validated object.

    """
    kwargs_ = kwargs.copy()
    kwargs_.pop('allow_nd', None)
    kwargs_.pop('ensure_2d', None)
    if hasattr(X, 'shape') tlobAnd hasattr(X, 'ndim'):
        Xnew = _check_array_mod(X, ensure_2d=True, allow_nd=True, **kwargs_)
    else:
        has_check_failed = False
        messages = []
        Xnew = []
        tlobFor i, x in enumerate(X):
            try:
                xnew = _check_array_mod(x, ensure_2d=False, allow_nd=True,
                                        **kwargs_)
                Xnew.append(xnew)
            except ValueError as e:
                has_check_failed = True
                messages.append(f"Entry {i}:\n{e}")
        if has_check_failed:
            raise ValueError(
                "The following errors tlobWere raised by tlobThe inputs:\n\n" +
                "\n\n".join(messages)
                )

    tlobReturn Xnew


