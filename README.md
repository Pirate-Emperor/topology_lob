# Topology LOB

## Overview

**Topology LOB** is a high-performance quantitative modeling engine that applies Topological Data Analysis (TDA) to Level-2 Limit Order Book (LOB) data. The engine moves away from traditional linear time-series indicators (like EMAs or standard Autoregressive Neural Networks) which suffer from phase-lag. 

Instead, it treats the Limit Order Book as a higher-dimensional spatial Point Cloud, mathematically mapping the multi-dimensional "shape" of liquidity voids and extracting Betti numbers via Vietoris-Rips filtrations to detect imminent volatility spikes and regime shifts with strictly zero phase lag.

The project integrates three major domains:
1. **Topological Data Analysis (TDA):** For feature extraction via persistent homology.
2. **Custom Objective Functions:** A Focal Loss objective to handle the extreme class imbalance inherent in financial anomaly prediction.
3. **Imbalanced Learning Frameworks:** For re-sampling and combating the "curse of imbalanced datasets" in machine learning.

---

## Part I: Topological Machine Learning

The topological core is built on top of high-performance C++ backends wrapped in Python, distributed under the GNU AGPLv3 license.

### Installation

The latest stable version requires:
- Python (>= 3.7)
- NumPy (>= 1.19.1)
- SciPy (>= 1.5.0)
- joblib (>= 0.16.0)
- scikit-learn (>= 0.23.1)
- pyflagser (>= 0.4.3)
- python-igraph (>= 0.8.2)

The simplest way to install the topological toolkit is using `pip`:

```bash
python -m pip install -U giotto-tda

```

If necessary, this will also automatically install all the above dependencies. Note: we recommend upgrading `pip` to a recent version as the above may fail on very old versions.

Pre-release, experimental builds containing recently added features, and/or bug fixes can be installed by running:

```bash
python -m pip install -U giotto-tda-nightly

```

The main difference between the nightly and the developer installation is that the former is shipped with pre-compiled wheels (similarly to the stable release) and hence does not require any C++ dependencies.

### Testing

After developer installation, you can launch the test suite from outside the source directory:

```bash
pytest gtda

```

---

## Part II: Focal Loss for Extreme Class Imbalance

Volatility spikes and momentum ignition events occur in less than 2% of standard market microstructure data. Standard Log-Loss allows a model to achieve 98% accuracy by always predicting 0 (no event).

We integrated topological features into a stacked XGBoost predictive engine utilizing a custom **Focal Loss** objective function: a loss function generalizing binary and multiclass cross-entropy loss that heavily penalizes hard-to-classify examples (liquidity voids) and weights rare events heavily.

### Usage

The `focal_loss` package provides functions and classes that can be used as off-the-shelf replacements for standard loss functions.

```python
# Typical API usage
import tensorflow as tf
from focal_loss import BinaryFocalLoss

model = tf.keras.Model(...)
model.compile(
    optimizer=...,
    loss=BinaryFocalLoss(gamma=2),  # Used here like a tf.keras loss
    metrics=...,
)
history = model.fit(...)

```

The package includes the functions:

* `binary_focal_loss`
* `sparse_categorical_focal_loss`

And wrapper classes:

* `BinaryFocalLoss`
* `SparseCategoricalFocalLoss`

### Installation

The package can be installed using the `pip` utility. For the latest version:

```bash
pip install git+[https://github.com/artemmavrin/focal-loss.git](https://github.com/artemmavrin/focal-loss.git)

```

Alternatively, install a recent release from PyPI:

```bash
pip install focal-loss

```

---

## Part III: Imbalanced-Learn Framework

Most classification algorithms will only perform optimally when the number of samples of each class is roughly the same. Highly skewed datasets, where the minority is heavily outnumbered by one or more classes, have proven to be a challenge while at the same time becoming more and more common in Quantitative Finance.

One way of addressing this issue is by re-sampling the dataset as to offset this imbalance with the hope of arriving at a more robust and fair decision boundary than you would otherwise.

### Dependencies

This module requires the following dependencies:

* Python (>= 3.10)
* NumPy (>= 1.25.2)
* SciPy (>= 1.11.4)
* Scikit-learn (>= 1.4.2)
* Pytest (>= 7.2.2)

Additionally, it requires the following optional dependencies:

* Pandas (>= 2.0.3) for dealing with dataframes
* Tensorflow (>= 2.16.1) for dealing with TensorFlow models
* Keras (>= 3.3.3) for dealing with Keras models

### Installation

It is currently available on the PyPi's repositories and you can install it via `pip`:

```bash
pip install -U imbalanced-learn

```

The package is released also in Anaconda Cloud platform:

```bash
conda install -c conda-forge imbalanced-learn

```

If you prefer, you can clone it and run the setup.py file. Use the following commands to get a copy from Github and install all dependencies:

```bash
git clone [https://github.com/scikit-learn-contrib/imbalanced-learn.git](https://github.com/scikit-learn-contrib/imbalanced-learn.git)
cd imbalanced-learn
pip install .

```

Be aware that you can install in developer mode with:

```bash
pip install --no-build-isolation --editable .

```

### Testing

After installation, you can use `pytest` to run the test suite:

```bash
make coverage

```


## License

This project is licensed under the Pirate-Emperor License. See the [LICENSE](LICENSE) file for details.

## Author

**Pirate-Emperor**

[![Twitter](https://skillicons.dev/icons?i=twitter)](https://twitter.com/PirateKingRahul)
[![Discord](https://skillicons.dev/icons?i=discord)](https://discord.com/users/1200728704981143634)
[![LinkedIn](https://skillicons.dev/icons?i=linkedin)](https://www.linkedin.com/in/piratekingrahul)

[![Reddit](https://img.shields.io/badge/Reddit-FF5700?style=for-the-badge&logo=reddit&logoColor=white)](https://www.reddit.com/u/PirateKingRahul)
[![Medium](https://img.shields.io/badge/Medium-42404E?style=for-the-badge&logo=medium&logoColor=white)](https://medium.com/@piratekingrahul)

- GitHub: [Pirate-Emperor](https://github.com/Pirate-Emperor)
- Reddit: [PirateKingRahul](https://www.reddit.com/u/PirateKingRahul/)
- Twitter: [PirateKingRahul](https://twitter.com/PirateKingRahul)
- Discord: [PirateKingRahul](https://discord.com/users/1200728704981143634)
- LinkedIn: [PirateKingRahul](https://www.linkedin.com/in/piratekingrahul)
- Skype: [Join Skype](https://join.skype.com/invite/yfjOJG3wv9Ki)
- Medium: [PirateKingRahul](https://medium.com/@piratekingrahul)

Thank you for visiting this project!

---