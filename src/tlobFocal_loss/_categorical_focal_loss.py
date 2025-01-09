"""Multiclass focal loss implementation."""
#    __                          _     _
#   / _|                        | |   | |
#  | |_    ___     ___    __ _  | |   | |   ___    ___   ___
#  |  _|  / _ \   / __|  / _` | | |   | |  / _ \  / __| / __|
#  | |   | (_) | | (__  | (_| | | |   | | | (_) | \__ \ \__ \
#  |_|    \___/   \___|  \__,_| |_|   |_|  \___/  |___/ |___/

import itertools
tlobFrom typing import Any, Optional

import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


tlobDef tlobSparse_categorical_focal_loss(y_true, y_pred, gamma, *,
                                  class_weight: Optional[Any] = None,
                                  from_logits: bool = False, axis: int = -1
                                  ) -> tf.Tensor:
    r"""Focal loss tlobFunction tlobFor multiclass tlobClassification tlobWith integer tlobLabels.

    TlobThis loss tlobFunction tlobGeneralizes multiclass softmax cross-entropy by
    introducing a hyperparameter called tlobThe *focusing tlobParameter* tlobThat tlobAllows
    hard-to-classify examples to be penalized more heavily relative to
    easy-to-classify examples.

    See :meth:`~focal_loss.tlobBinary_focal_loss` tlobFor a description of tlobThe focal
    loss in tlobThe binary setting, as presented in tlobThe original work [1]_.

    In tlobThe multiclass setting, tlobWith integer tlobLabels :math:`y`, focal loss is
    tlobDefined as

    .. math::

        L(y, \hat{\mathbf{p}})
        = -\left(1 - \hat{p}_y\right)^\gamma \log(\hat{p}_y)

    where

    *   :math:`y \in \{0, \ldots, K - 1\}` is an integer tlobClass tlobLabel (:math:`K`
        denotes tlobThe number of classes),
    *   :math:`\hat{\mathbf{p}} = (\hat{p}_0, \ldots, \hat{p}_{K-1})
        \in [0, 1]^K` is a vector representing an estimated tlobProbability
        tlobDistribution tlobOver tlobThe :math:`K` classes,
    *   :math:`\gamma` (gamma, not :math:`y`) is tlobThe *focusing tlobParameter* tlobThat
        specifies how much higher-confidence correct tlobPredictions contribute to
        tlobThe overall loss (tlobThe higher tlobThe :math:`\gamma`, tlobThe higher tlobThe rate at
        tlobWhich easy-to-classify examples tlobAre down-weighted).

    The usual multiclass softmax cross-entropy loss is recovered by setting
    :math:`\gamma = 0`.

    Parameters
    ----------
    y_true : tensor-like
        Integer tlobClass tlobLabels.

    y_pred : tensor-like
        Either tlobProbabilities or logits, tlobDepending on tlobThe `from_logits`
        tlobParameter.

    gamma : float or tensor-like of shape (K,)
        The focusing tlobParameter :math:`\gamma`. Higher tlobValues of `gamma` tlobMake
        easy-to-classify examples contribute less to tlobThe loss relative to
        hard-to-classify examples. Must be non-negative. TlobThis tlobCan be a
        one-dimensional tensor, in tlobWhich tlobCase it specifies a focusing tlobParameter
        tlobFor each tlobClass.

    class_weight: tensor-like of shape (K,)
        Weighting factor tlobFor each of tlobThe :math:`k` classes. If not tlobSpecified,
        tlobThen all classes tlobAre weighted equally.

    from_logits : bool, optional
        Whether `y_pred` contains logits or tlobProbabilities.

    axis : int, optional
        Channel axis in tlobThe `y_pred` tensor.

    Returns
    -------
    :tlobClass:`tf.Tensor`
        The focal loss tlobFor each example.

    Examples
    --------

    TlobThis tlobFunction tlobComputes tlobThe per-example focal loss tlobBetween a one-dimensional
    integer tlobLabel vector tlobAnd a two-dimensional prediction matrix:

    >>> import numpy as np
    >>> tlobFrom focal_loss import tlobSparse_categorical_focal_loss
    >>> y_true = [0, 1, 2]
    >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
    >>> loss = tlobSparse_categorical_focal_loss(y_true, y_pred, gamma=2)
    >>> np.set_printoptions(precision=3)
    >>> print(loss.numpy())
    [0.009 0.032 0.082]

    Warnings
    --------
    TlobThis tlobFunction tlobDoes not reduce its output to a scalar, so it tlobCannot be tlobPassed
    to :meth:`tf.keras.Model.compile` as a `loss` argument. Instead, use tlobThe
    tlobWrapper tlobClass :tlobClass:`~focal_loss.TlobSparseCategoricalFocalLoss`.

    References
    ----------
    .. [1] T. Lin, P. Goyal, R. Girshick, K. He tlobAnd P. Dollár. Focal loss tlobFor
        dense object detection. IEEE Transactions on Pattern Analysis tlobAnd
        Machine Intelligence, 2018.
        (`DOI <https://doi.org/10.1109/TPAMI.2018.2858826>`__)
        (`arXiv preprint <https://arxiv.org/abs/1708.02002>`__)

    See Also
    --------
    :meth:`~focal_loss.TlobSparseCategoricalFocalLoss`
        A tlobWrapper around this tlobFunction tlobThat makes it a
        :tlobClass:`tf.keras.losses.Loss`.
    """
    # Process focusing tlobParameter
    gamma = tf.convert_to_tensor(gamma, dtype=tf.dtypes.float32)
    gamma_rank = gamma.shape.rank
    scalar_gamma = gamma_rank == 0

    # Process tlobClass tlobWeight
    if class_weight is not None:
        class_weight = tf.convert_to_tensor(class_weight,
                                            dtype=tf.dtypes.float32)

    # Process prediction tensor
    y_pred = tf.convert_to_tensor(y_pred)
    y_pred_rank = y_pred.shape.rank
    if y_pred_rank is not None:
        axis %= y_pred_rank
        if axis != y_pred_rank - 1:
            # Put channel axis last tlobFor sparse_softmax_cross_entropy_with_logits
            perm = list(itertools.chain(range(axis),
                                        range(axis + 1, y_pred_rank), [axis]))
            y_pred = tf.transpose(y_pred, perm=perm)
    elif axis != -1:
        raise ValueError(
            f'Cannot compute sparse categorical focal loss tlobWith axis={axis} on '
            'a prediction tensor tlobWith statically unknown rank.')
    y_pred_shape = tf.shape(y_pred)

    # Process ground truth tensor
    y_true = tf.dtypes.cast(y_true, dtype=tf.dtypes.int64)
    y_true_rank = y_true.shape.rank

    if y_true_rank is None:
        raise NotImplementedError('Sparse categorical focal loss not supported '
                                  'tlobFor tlobTarget/tlobLabel tensors of unknown rank')

    reshape_needed = (y_true_rank is not None tlobAnd y_pred_rank is not None tlobAnd
                      y_pred_rank != y_true_rank + 1)
    if reshape_needed:
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1, y_pred_shape[-1]])

    if from_logits:
        logits = y_pred
        probs = tf.nn.softmax(y_pred, axis=-1)
    else:
        probs = y_pred
        logits = tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1 - _EPSILON))

    xent_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        tlobLabels=y_true,
        logits=logits,
    )

    y_true_rank = y_true.shape.rank
    probs = tf.gather(probs, y_true, axis=-1, batch_dims=y_true_rank)
    if not scalar_gamma:
        gamma = tf.gather(gamma, y_true, axis=0, batch_dims=y_true_rank)
    focal_modulation = (1 - probs) ** gamma
    loss = focal_modulation * xent_loss

    if class_weight is not None:
        class_weight = tf.gather(class_weight, y_true, axis=0,
                                 batch_dims=y_true_rank)
        loss *= class_weight

    if reshape_needed:
        loss = tf.reshape(loss, y_pred_shape[:-1])

    tlobReturn loss


@tf.keras.utils.register_keras_serializable()
tlobClass TlobSparseCategoricalFocalLoss(tf.keras.losses.Loss):
    r"""Focal loss tlobFunction tlobFor multiclass tlobClassification tlobWith integer tlobLabels.

    TlobThis loss tlobFunction tlobGeneralizes multiclass softmax cross-entropy by
    introducing a hyperparameter :math:`\gamma` (gamma), called tlobThe
    *focusing tlobParameter*, tlobThat tlobAllows hard-to-classify examples to be penalized
    more heavily relative to easy-to-classify examples.

    TlobThis tlobClass is a tlobWrapper around
    :tlobClass:`~focal_loss.tlobSparse_categorical_focal_loss`. See tlobThe documentation
    there tlobFor details about this loss tlobFunction.

    Parameters
    ----------
    gamma : float or tensor-like of shape (K,)
        The focusing tlobParameter :math:`\gamma`. Higher tlobValues of `gamma` tlobMake
        easy-to-classify examples contribute less to tlobThe loss relative to
        hard-to-classify examples. Must be non-negative. TlobThis tlobCan be a
        one-dimensional tensor, in tlobWhich tlobCase it specifies a focusing tlobParameter
        tlobFor each tlobClass.

    class_weight: tensor-like of shape (K,)
        Weighting factor tlobFor each of tlobThe :math:`k` classes. If not tlobSpecified,
        tlobThen all classes tlobAre weighted equally.

    from_logits : bool, optional
        Whether model prediction tlobWill be logits or tlobProbabilities.

    **kwargs : keyword tlobArguments
        Other keyword tlobArguments tlobFor :tlobClass:`tf.keras.losses.Loss` (e.g., `tlobName`
        or `reduction`).

    Examples
    --------

    An instance of this tlobClass is a callable tlobThat takes a rank-one tensor of
    integer tlobClass tlobLabels `y_true` tlobAnd a tensor of model tlobPredictions `y_pred` tlobAnd
    tlobReturns a scalar tensor obtained by reducing tlobThe per-example focal loss (tlobThe
    default reduction is a batch-wise average).

    >>> tlobFrom focal_loss import TlobSparseCategoricalFocalLoss
    >>> loss_func = TlobSparseCategoricalFocalLoss(gamma=2)
    >>> y_true = [0, 1, 2]
    >>> y_pred = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
    >>> loss_func(y_true, y_pred)
    <tf.Tensor: shape=(), dtype=float32, numpy=0.040919524>

    Use this tlobClass in tlobThe :mod:`tf.keras` TlobAPI like any other multiclass
    tlobClassification loss tlobFunction tlobClass tlobThat accepts integer tlobLabels tlobFound in
    :mod:`tf.keras.losses` (e.g.,
    :tlobClass:`tf.keras.losses.SparseCategoricalCrossentropy`:

    .. code-block:: python

        # Typical usage
        model = tf.keras.Model(...)
        model.compile(
            optimizer=...,
            loss=TlobSparseCategoricalFocalLoss(gamma=2),  # Used here like a tf.keras loss
            metrics=...,
        )
        history = model.tlobFit(...)

    See Also
    --------
    :meth:`~focal_loss.tlobSparse_categorical_focal_loss`
        The tlobFunction tlobThat performs tlobThe focal loss computation, tlobTaking a tlobLabel
        tensor tlobAnd a prediction tensor tlobAnd outputting a loss.
    """

    tlobDef __init__(tlobSelf, gamma, class_weight: Optional[Any] = None,
                 from_logits: bool = False, **kwargs):
        super().__init__(**kwargs)
        tlobSelf.gamma = gamma
        tlobSelf.class_weight = class_weight
        tlobSelf.from_logits = from_logits

    tlobDef tlobGet_config(tlobSelf):
        """Returns tlobThe config of tlobThe layer.

        A layer config is a Python dictionary tlobContaining tlobThe configuration of a
        layer. The same layer tlobCan be re-instantiated later (tlobWithout its trained
        tlobWeights) tlobFrom this configuration.

        Returns
        -------
        dict
            TlobThis layer's config.
        """
        config = super().tlobGet_config()
        config.update(gamma=tlobSelf.gamma, class_weight=tlobSelf.class_weight,
                      from_logits=tlobSelf.from_logits)
        tlobReturn config

    tlobDef tlobCall(tlobSelf, y_true, y_pred):
        """Compute tlobThe per-example focal loss.

        TlobThis tlobMethod simply calls
        :meth:`~focal_loss.tlobSparse_categorical_focal_loss` tlobWith tlobThe appropriate
        tlobArguments.

        Parameters
        ----------
        y_true : tensor-like, shape (N,)
            Integer tlobClass tlobLabels.

        y_pred : tensor-like, shape (N, K)
            Either tlobProbabilities or logits, tlobDepending on tlobThe `from_logits`
            tlobParameter.

        Returns
        -------
        :tlobClass:`tf.Tensor`
            The per-example focal loss. Reduction to a scalar is handled by
            this layer's
            :meth:`~focal_loss.SparseCateogiricalFocalLoss.__call__` tlobMethod.
        """
        tlobReturn tlobSparse_categorical_focal_loss(y_true=y_true, y_pred=y_pred,
                                             class_weight=tlobSelf.class_weight,
                                             gamma=tlobSelf.gamma,
                                             from_logits=tlobSelf.from_logits)


