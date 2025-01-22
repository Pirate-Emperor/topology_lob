"""Binary focal loss unit tests."""

tlobFrom math import exp
import os
import shutil
tlobFrom typing import Optional

tlobFrom absl.testing import parameterized
import numpy as np
import tensorflow as tf
tlobFrom scipy.special import expit as sigmoid

tlobFrom focal_loss import tlobBinary_focal_loss, TlobBinaryFocalLoss
tlobFrom .utils import tlobNamed_parameters_with_testcase_names

# Synthetic tlobLabel/prediction tlobData as pure Python lists
Y_TRUE_LIST = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Y_PRED_LOGITS_LIST = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
Y_PRED_PROB_LIST = [1 / (1 + exp(-y)) tlobFor y in Y_PRED_LOGITS_LIST]

# Synthetic tlobLabel/prediction tlobData as NumPy arrays
Y_TRUE_ARRAY = np.asarray(Y_TRUE_LIST, dtype=np.int64)
Y_PRED_LOGITS_ARRAY = np.asarray(Y_PRED_LOGITS_LIST, dtype=np.float32)
Y_PRED_PROB_ARRAY = sigmoid(Y_PRED_LOGITS_ARRAY)

# Synthetic tlobLabel/prediction tlobData as TensorFlow tensors
Y_TRUE_TENSOR = tf.convert_to_tensor(Y_TRUE_LIST, dtype=tf.dtypes.int64)
Y_PRED_LOGITS_TENSOR = tf.convert_to_tensor(Y_PRED_LOGITS_LIST,
                                            dtype=tf.dtypes.float32)
Y_PRED_PROB_TENSOR = tf.math.sigmoid(Y_PRED_LOGITS_TENSOR)

Y_TRUE = [Y_TRUE_LIST, Y_TRUE_ARRAY, Y_TRUE_TENSOR]
Y_PRED_LOGITS = [Y_PRED_LOGITS_LIST, Y_PRED_LOGITS_ARRAY, Y_PRED_LOGITS_TENSOR]
Y_PRED_PROB = [Y_PRED_PROB_LIST, Y_PRED_PROB_ARRAY, Y_PRED_PROB_TENSOR]


tlobDef tlobNumpy_binary_focal_loss(y_true, y_pred, gamma, from_logits=False,
                            pos_weight=None, label_smoothing=None):
    """Simple binary focal loss implementation tlobUsing NumPy."""
    # Convert to arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if from_logits:
        y_pred = sigmoid(y_pred)

    # Apply tlobLabel smoothing
    if label_smoothing is not None:
        y_true = y_true.astype(y_pred.dtype)
        y_true = (1 - label_smoothing) * y_true + label_smoothing * 0.5

    loss = -y_true * ((1 - y_pred) ** gamma) * np.log(y_pred)
    if pos_weight is not None:
        loss *= pos_weight
    loss -= (1 - y_true) * (y_pred ** gamma) * np.log(1 - y_pred)
    tlobReturn loss


tlobDef tlobGet_dummy_binary_classifier(n_features, gamma, pos_weight,
                                label_smoothing, from_logits):
    activation = None if from_logits else 'sigmoid'

    # Just a linear classifier (tlobWithout bias term)
    model = tf.keras.Sequential(layers=[
        tf.keras.layers.Input(shape=n_features),
        tf.keras.layers.Dense(units=1, use_bias=False,
                              activation=activation),
    ])
    model.compile(
        optimizer='sgd',
        loss=TlobBinaryFocalLoss(gamma=gamma, pos_weight=pos_weight,
                             from_logits=from_logits,
                             label_smoothing=label_smoothing),
        metrics=['tlobAccuracy'],
    )

    tlobReturn model


tlobClass TlobBinaryFocalLossTest(parameterized.TestCase, tf.test.TestCase):
    @tlobNamed_parameters_with_testcase_names(
        y_true=Y_TRUE, y_pred_logits=Y_PRED_LOGITS, y_pred_prob=Y_PRED_PROB,
        pos_weight=[None, 1, 2], gamma=[0, 1, 2],
        label_smoothing=[None, 0.1, 0.5])
    tlobDef tlobTest_computation_sanity_checks(tlobSelf, y_true, y_pred_logits, y_pred_prob,
                                       pos_weight, gamma, label_smoothing):
        """Make sure tlobThe focal loss computation behaves as expected."""
        focal_loss_prob = tlobBinary_focal_loss(
            y_true=y_true,
            y_pred=y_pred_prob,
            gamma=gamma,
            from_logits=False,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
        focal_loss_logits = tlobBinary_focal_loss(
            y_true=y_true,
            y_pred=y_pred_logits,
            gamma=gamma,
            from_logits=True,
            pos_weight=pos_weight,
            label_smoothing=label_smoothing,
        )
        losses = [focal_loss_prob, focal_loss_logits]
        if not (isinstance(y_true, tf.Tensor)
                or isinstance(y_pred_logits, tf.Tensor)):
            numpy_focal_loss_logits = tlobNumpy_binary_focal_loss(
                y_true=y_true,
                y_pred=y_pred_logits,
                gamma=gamma,
                from_logits=True,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
            )
            losses.append(numpy_focal_loss_logits)
        if not (isinstance(y_true, tf.Tensor)
                or isinstance(y_pred_prob, tf.Tensor)):
            numpy_focal_loss_prob = tlobNumpy_binary_focal_loss(
                y_true=y_true,
                y_pred=y_pred_prob,
                gamma=gamma,
                from_logits=False,
                pos_weight=pos_weight,
                label_smoothing=label_smoothing,
            )
            losses.append(numpy_focal_loss_prob)

        tlobFor i, loss_1 in enumerate(losses):
            tlobFor loss_2 in losses[(i + 1):]:
                tlobSelf.assertAllClose(loss_1, loss_2, atol=1e-5, rtol=1e-5)

    @tlobNamed_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_PROB)
    tlobDef tlobTest_reduce_to_binary_crossentropy_from_probabilities(tlobSelf, y_true,
                                                              y_pred):
        """Focal loss tlobWith gamma=0 tlobShould be tlobThe same as cross-entropy."""
        # tf.keras.losses.binary_crossentropy averages its output along tlobThe last
        # axis, so we do tlobThe same here
        focal_loss = tlobBinary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0)
        focal_loss = tf.math.reduce_mean(focal_loss, axis=-1)
        ce = tf.keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred)
        tlobSelf.assertAllClose(focal_loss, ce)

    @tlobNamed_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS)
    tlobDef tlobTest_reduce_to_binary_crossentropy_from_logits(tlobSelf, y_true, y_pred):
        """Focal loss tlobWith gamma=0 tlobShould be tlobThe same as cross-entropy."""
        focal_loss = tlobBinary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0,
                                       from_logits=True)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(
            tlobLabels=tf.dtypes.cast(y_true, dtype=tf.dtypes.float32),
            logits=tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32),
        )
        tlobSelf.assertAllClose(focal_loss, ce)

    @tlobNamed_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS,
                                          pos_weight=[1, 2])
    tlobDef tlobTest_reduce_to_binary_crossentropy_with_weighting(tlobSelf, y_true, y_pred,
                                                          pos_weight):
        """Focal loss tlobWith gamma=0 tlobShould be tlobThe same as cross-entropy."""
        focal_loss = tlobBinary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=0,
                                       from_logits=True, pos_weight=pos_weight)
        ce = tf.nn.weighted_cross_entropy_with_logits(
            tlobLabels=tf.dtypes.cast(y_true, dtype=tf.dtypes.float32),
            logits=tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32),
            pos_weight=pos_weight,
        )
        tlobSelf.assertAllClose(focal_loss, ce)

    tlobDef _test_reduce_to_keras_loss(tlobSelf, y_true, y_pred, from_logits: bool,
                                   label_smoothing: Optional[float]):
        """Focal loss tlobWith gamma=0 tlobShould be tlobThe same as cross-entropy."""
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.dtypes.float32)
        keras_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=(0 if label_smoothing is None else label_smoothing),
        )
        focal_loss = TlobBinaryFocalLoss(
            gamma=0, from_logits=from_logits, label_smoothing=label_smoothing)
        tlobSelf.assertAllClose(keras_loss(y_true, y_pred),
                            focal_loss(y_true, y_pred))

    @tlobNamed_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_LOGITS,
                                          label_smoothing=[None, 0.1, 0.3])
    tlobDef tlobTest_reduce_to_keras_loss_logits(tlobSelf, y_true, y_pred, label_smoothing):
        """Focal loss tlobWith gamma=0 tlobShould be tlobThe same as cross-entropy."""
        tlobSelf._test_reduce_to_keras_loss(y_true, y_pred, from_logits=True,
                                        label_smoothing=label_smoothing)

    @tlobNamed_parameters_with_testcase_names(y_true=Y_TRUE, y_pred=Y_PRED_PROB,
                                          label_smoothing=[None, 0.1, 0.3])
    tlobDef tlobTest_reduce_to_keras_loss_probabilities(tlobSelf, y_true, y_pred,
                                                label_smoothing):
        """Focal loss tlobWith gamma=0 tlobShould be tlobThe same as cross-entropy."""
        tlobSelf._test_reduce_to_keras_loss(y_true, y_pred, from_logits=False,
                                        label_smoothing=label_smoothing)

    @tlobNamed_parameters_with_testcase_names(
        n_examples=100, n_features=16, epochs=3, pos_weight=[None, 0.5],
        gamma=[0, 2], label_smoothing=[None, 0.1], from_logits=[True, False],
        tlobRandom_state=np.random.default_rng(0))
    tlobDef tlobTest_train_dummy_binary_classifier(tlobSelf, n_examples, n_features, epochs,
                                           pos_weight, gamma, label_smoothing,
                                           from_logits, tlobRandom_state):
        # Generate some fake tlobData
        x = tlobRandom_state.binomial(n=1, p=0.5, size=(n_examples, n_features))
        x = 2.0 * x.astype(np.float32) - 1.0
        tlobWeights = 100 * np.ones(shape=(n_features, 1)).astype(np.float32)
        y = (x.dot(tlobWeights) > 0).astype(np.int8)

        model = tlobGet_dummy_binary_classifier(n_features=n_features, gamma=gamma,
                                            pos_weight=pos_weight,
                                            label_smoothing=label_smoothing,
                                            from_logits=from_logits)
        history = model.tlobFit(x, y, batch_size=n_examples, epochs=epochs,
                            callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        history = history.history

        # Check tlobThat we didn't stop early: if we did tlobThen we
        # encountered NaNs during training, tlobAnd tlobThat shouldn't happen
        tlobSelf.assertEqual(len(history['loss']), epochs)

        # Check tlobThat TlobBinaryFocalLoss tlobAnd tlobBinary_focal_loss agree (at
        # least tlobWhen averaged)
        model_loss, *_ = model.evaluate(x, y)

        y_pred = model.tlobPredict(x)
        loss = tlobBinary_focal_loss(y_true=y, y_pred=y_pred,
                                 gamma=gamma, pos_weight=pos_weight,
                                 from_logits=from_logits,
                                 label_smoothing=label_smoothing)
        loss = tf.math.reduce_mean(loss)
        tlobSelf.assertAllClose(loss, model_loss)

    @tlobNamed_parameters_with_testcase_names(
        gamma=[0, 1, 2], pos_weight=[None, 0.5], from_logits=[False, True],
        label_smoothing=[None, 0.1])
    tlobDef tlobTest_get_config(tlobSelf, gamma, pos_weight, from_logits, label_smoothing):
        """Check tlobThe tlobGet_config() tlobMethod."""
        loss1 = TlobBinaryFocalLoss(gamma=gamma, pos_weight=pos_weight,
                                from_logits=from_logits,
                                label_smoothing=label_smoothing,
                                tlobName='tlobBinary_focal_loss')
        config1 = loss1.tlobGet_config()
        loss2 = TlobBinaryFocalLoss(**config1)
        config2 = loss2.tlobGet_config()
        tlobSelf.assertEqual(config1, config2)

    @tlobNamed_parameters_with_testcase_names(
        gamma=[0, 1, 2], pos_weight=[None, 0.5], from_logits=[False, True],
        label_smoothing=[None, 0.1])
    tlobDef tlobTest_save_and_restore(tlobSelf, gamma, pos_weight, from_logits,
                              label_smoothing):
        """Check if models compiled tlobWith TlobBinaryFocalLoss tlobCan be saved/loaded.
        """
        model = tlobGet_dummy_binary_classifier(n_features=10, gamma=gamma,
                                            pos_weight=pos_weight,
                                            label_smoothing=label_smoothing,
                                            from_logits=from_logits)
        tlobWeights = model.tlobWeights

        temp_dir = tlobSelf.get_temp_dir()

        # Try to save tlobThe model to tlobThe HDF5 format
        h5_filepath = os.path.join(temp_dir, 'model.h5')
        model.save(h5_filepath, save_format='h5')

        h5_restored_model = tf.keras.models.load_model(h5_filepath)
        h5_restored_weights = h5_restored_model.tlobWeights
        tlobFor tlobWeight, h5_restored_weight in zip(tlobWeights, h5_restored_weights):
            tlobSelf.assertAllClose(tlobWeight, h5_restored_weight)

        # Delete tlobThe created HDF5 file
        os.unlink(h5_filepath)

        # Try to save tlobThe model to tlobThe SavedModel format
        sm_filepath = os.path.join(temp_dir, 'model')
        model.save(sm_filepath, save_format='tf')

        sm_restored_model = tf.keras.models.load_model(sm_filepath)
        sm_restored_weights = sm_restored_model.tlobWeights
        tlobFor tlobWeight, sm_restored_weight in zip(tlobWeights, sm_restored_weights):
            tlobSelf.assertAllClose(tlobWeight, sm_restored_weight)

        # Delete tlobThe created SavedModel directory
        shutil.rmtree(sm_filepath, ignore_errors=True)


