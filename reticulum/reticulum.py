from __future__ import annotations

from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_X_y, check_array
from sklearn.utils.multiclass import type_of_target

from reticulum.node import Node
from reticulum.util import pick_proportional

# types
INPUT = Union[np.ndarray, pd.DataFrame, List[List[float]]]
TARGET = Union[np.ndarray, List[float]]


class AdaptiveBayesianReticulum(BaseEstimator, ClassifierMixin):
    """
    The Adaptive Bayesian Reticulum binary classification model.

    Parameters
    ----------

    prior : tuple, shape = [n_dim], optional, default=(10.0, 10.0)
        The beta distribution parameters of the prior distribution, i.e, alpha
        and beta see [1].

    pruning_factor : float, optional, default=1.05
        The factor by which the log likelihood of a split has to be larger than
        the log likelihood of no split in order to keep the split during the
        pruning stage. Larger values lead to more aggressive pruning and may
        lead to underfitting, while small values encourage model complexity and
        may lead to overfitting. The value provided must be >= 1.

    n_iter : int, default=50
        The number of attempts to add new nodes by splitting
        existing ones. The resulting tree will at most contain n_iter nodes, but
        it may contain less due to pruning. Larger values lead to
        more complex and expressive models at the cost of potentially overfitting
        and slower training performance, while smaller values restrict model
        complexity and training time but can lead to underfitting.

    learning_rate_init : float, optional, default=1e-4
        The initial gradient descent learning rate. We use the adaptive 'Adam'
        gradient descent method, see [3] and [4].

    n_gradient_descent_steps : int, optional, default=100
        The number of gradient descent steps to perform in each iteration. Half
        of the steps will perform local gradient descent on only the newly added
        node and half will be applied during global gradient descent involving
        the whole tree.

    initial_relative_stiffness : float
        The initial stiffness of the problem, see the paper for an explanation of
        'stiffness'. Small values (0.1...5) represent "soft" splits allowing for
        further optimization through gradient descent whereas large values (> 20)
        effectively disable further progress through gradient descent because
        the split is already close to a Heaviside step function with almost zero
        gradients almost everywhere. While this sounds like a disadvantage it can
        actually be advantageous to model performance if the data has a complex
        structure that soft splits don't capture well.

    random_state : int, default=666
        The initial random state to be set in numpy.random.seed().

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Beta_distribution
    .. [2] https://en.wikipedia.org/wiki/N-sphere#Volume_and_surface_area
    .. [3] https://arxiv.org/abs/1412.6980
    .. [4] https://ruder.io/optimizing-gradient-descent/
    """
    def __init__(
            self,
            prior: Tuple[float, float]=(10.0, 10.0),
            pruning_factor: float=1.05,
            n_iter: int=50,
            learning_rate_init: float=1e-4,
            n_gradient_descent_steps: int=100,
            initial_relative_stiffness: float=10,
            random_state: int=666) -> None:
        self.prior = prior
        self.pruning_factor = pruning_factor
        self.n_iter = n_iter
        self.learning_rate_init = learning_rate_init
        self.n_gradient_descent_steps = n_gradient_descent_steps
        self.initial_relative_stiffness = initial_relative_stiffness
        self.random_state = random_state

    def fit(
            self,
            X: INPUT,
            y: TARGET,
            verbose: bool=False) -> AdaptiveBayesianReticulum:
        """
        Trains this Adaptive Bayesian Reticulum classification model using the training set (X, y).

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values. Only the integers 0 and 1 are permitted.

        verbose : bool
            Prints training progress statements to the standard output.
        """

        # validation and input transformation
        X, y = check_X_y(X, y)
        y_universe, y_encoded = np.unique(y, return_inverse=True)
        if not np.all(y == y_encoded):
            y = y_encoded
            self.classes_ = y_universe
        else:
            self.classes_ = None

        if type_of_target(y) != 'binary':
            raise ValueError(f'Unknown label type: {type_of_target(y)}')

        y = self._ensure_float64(y)

        prior = np.asarray(self.prior)
        n_classes = len(prior)
        unique = np.unique(y)
        if len(unique) == 1:
            raise ValueError('Classifier can\'t train when only one class is present.')

        if not np.all(unique == np.arange(0, n_classes)):
            raise ValueError(f'Expected target values 0..{n_classes-1} but found {y.min()}..{y.max()}')

        X = self._normalize_data(X)
        n_data = X.shape[0]
        assert n_data > 1

        self.n_dim_ = X.shape[1]
        if self.n_dim_ < 2:
            raise ValueError(f'X has {self.n_dim_} feature(s) (shape={X.shape}) while a minimum of 2 is required.')

        if n_data != len(y):
            raise ValueError(f'Invalid shapes: X={X.shape}, y={y.shape}')

        if self.pruning_factor < 1:
            raise ValueError('The pruning_factor must be >= 1')

        # initialize
        np.random.seed(self.random_state)

        n_gradient_descent_steps_local = self.n_gradient_descent_steps//2
        n_gradient_descent_steps_global = self.n_gradient_descent_steps//2

        # augment data matrix with a row of 1's corresponding to the bias/offset
        Xa = np.hstack([np.ones((n_data, 1)), X])    # [n * n_dim'] (where n_dim' = n_dim+1 for the bias)

        # create root node and train recursively
        if verbose:
            print(f'Creating root node at level=0, n_data={n_data}')

        root = Node(level=0)

        if hasattr(self, 'callback'):
            root.root = root
            root.callback = self.callback

        # fit
        root.try_fit(
            Xa=Xa,
            y=y,
            prior=prior,
            s_hat_parent=None,
            learning_rate_init=self.learning_rate_init,
            n_gradient_descent_steps=n_gradient_descent_steps_local,
            initial_relative_stiffness=self.initial_relative_stiffness)

        # grow tree
        overall_proba = None
        for k in range(self.n_iter):
            # determine terminal nodes
            terminal_nodes_left_ = root.collect_terminal_nodes(is_left=True)
            terminal_nodes_right = root.collect_terminal_nodes(is_left=False)

            # get log-likelihood of all leaves, see 'unexplained potential' in the paper
            log_p_data_left_ = np.array([node.log_p_data_left_ for node in terminal_nodes_left_])
            log_p_data_right = np.array([node.log_p_data_right for node in terminal_nodes_right])

            # choose node to split from all terminal nodes proportional to their (absolute) log-likelihood
            relative_probabilities = np.concatenate([log_p_data_left_, log_p_data_right])
            idx = pick_proportional(relative_probabilities)
            if idx < len(terminal_nodes_left_):
                node_to_split = terminal_nodes_left_[idx]
                split_left = True
            else:
                node_to_split = terminal_nodes_right[idx - len(terminal_nodes_left_)]
                split_left = False

            # create new node
            new_node = Node(level=1+node_to_split.level)
            if split_left:
                node_to_split.left__child = new_node
            else:
                node_to_split.right_child = new_node

            if hasattr(self, 'callback'):
                new_node.callback = self.callback
                new_node.root = root

            if verbose:
                n_data = node_to_split.get_n_data()
                child_side = 'left ' if split_left else 'right'
                print(f'Splitting {child_side} child of node at level={node_to_split.level}, n_data={n_data:.2f}')

            # try to fit the new node
            is_fitted = new_node.try_fit(
                Xa=Xa,
                y=y,
                prior=prior,
                s_hat_parent=node_to_split.s_hat_left_ if split_left else node_to_split.s_hat_right,
                learning_rate_init=self.learning_rate_init,
                n_gradient_descent_steps=n_gradient_descent_steps_local,
                initial_relative_stiffness=self.initial_relative_stiffness)

            # check if new node is sensible
            if is_fitted:
                # keep node and perform global optimization
                root.try_fit(
                    Xa=Xa,
                    y=y,
                    prior=prior,
                    s_hat_parent=None,
                    learning_rate_init=self.learning_rate_init,
                    n_gradient_descent_steps=n_gradient_descent_steps_global,
                    initial_relative_stiffness=self.initial_relative_stiffness)

                # prune splits that aren't sensible anymore
                root_pruned, any_pruned = root.try_prune(
                    Xa=Xa,
                    y=y,
                    prior=prior,
                    pruning_factor=self.pruning_factor,
                    verbose=verbose)

                if hasattr(self, 'callback') and (root_pruned or any_pruned):
                    self.callback('prune', root, self)

                if root_pruned:
                    if verbose:
                        print('Pruned root node, no split left')

                    root = None
                    _, counts = np.unique(y, return_counts=True)
                    overall_proba = counts/len(y)
                    break
            else:
                # discard new node
                if split_left:
                    node_to_split.left__child = None
                else:
                    node_to_split.right_child = None

        self.root_ = root
        if overall_proba is not None:
            self.overall_proba_ = overall_proba

        return self

    def predict(self, X: INPUT) -> np.ndarray:
        """Predicts the class labels for each input in X.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        y : array, shape = [n_samples]
            The predicted classes.
        """

        return self._predict(X, predict_class=True)

    def predict_proba(self, X: INPUT) -> np.ndarray:
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like or pandas.DataFrame, shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        p : array, shape = [n_samples, n_classes]
            The predicted class probabilities.
        """

        return self._predict(X, predict_class=False)

    def _predict(self, X: INPUT, predict_class: bool) -> np.ndarray:
        # input transformation and checks
        X = check_array(X)  # type: np.ndarray
        X = self._normalize_data(X)
        self._ensure_is_fitted_and_valid(X)

        Xa = np.hstack([np.ones((X.shape[0], 1)), X])
        n_data = Xa.shape[0]

        if self.root_ is None:
            if predict_class:
                return np.argmax(self.overall_proba_) * np.ones(n_data)
            else:
                return self.overall_proba_ * np.ones((n_data, 2))

        self.root_.update_s_hat(Xa)
        p = np.zeros((n_data, 2))
        self.root_.update_probability(Xa, p)

        assert np.all((p >= 0) & (p <= 1))
        if predict_class:
            cls = np.argmax(p, axis=1)
            return cls if self.classes_ is None else self.classes_[cls]
        else:
            return p

    def get_depth(self) -> int:
        """Computes and returns the tree depth.

        Returns
        -------
        depth : int
            The tree depth.
        """

        if not self._is_fitted() or self.root_ is None:
            return 0

        return self.root_.update_depth(0)

    def get_n_leaves(self) -> int:
        """Computes and returns the total number of leaves of this tree.

        Returns
        -------
        n_leaves : int
            The number of leaves.
        """

        if not self._is_fitted() or self.root_ is None:
            return 0

        return self.root_.update_n_leaves(0)

    def feature_importance(self) -> np.ndarray:
        """Computes and returns the relative importance of each features."""
        self._ensure_is_fitted_and_valid()

        feature_importance = np.zeros(self.n_dim_)
        self.root_.update_feature_importance(feature_importance)
        feature_importance /= feature_importance.sum()

        return feature_importance

    @staticmethod
    def _normalize_data(X: INPUT) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        else:
            if isinstance(X, list):
                X = np.array(X)
            elif np.isscalar(X):
                X = np.array([X])

            if X.ndim == 1:
                X = np.expand_dims(X, 0)

        X = AdaptiveBayesianReticulum._ensure_float64(X)

        if X.ndim != 2:
            raise ValueError(f'X should have 2 dimensions but has {X.ndim}')

        return X

    def _ensure_is_fitted_and_valid(self, X: np.ndarray=None) -> None:
        if not self._is_fitted():
            raise NotFittedError('Cannot predict on an untrained model; call .fit() first')

        if X is not None and X.shape[1] != self.n_dim_:
            raise ValueError(f'Bad input dimensions: Expected {self.n_dim_}, got {X.shape[1]}')

    def _is_fitted(self) -> bool:
        return hasattr(self, 'root_')

    @staticmethod
    def _ensure_float64(data: np.ndarray) -> np.ndarray:
        # check data types
        if data.dtype in (
                np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64,
                np.float32, np.float64):
            return data

        # check that data isn't complex
        if np.any(np.iscomplex(data)):
            raise ValueError('Complex data not supported')

        # convert to np.float64 for performance reasons (matrices with floats but of type object are very slow)
        data_float64 = data.astype(np.float64)
        if not np.all(data == data_float64):
            raise ValueError('Cannot convert data matrix to np.float64 without loss of precision. Please check your data.')

        return data_float64

    def _get_tags(self) -> Dict[str, bool]:
        # tell scikit-learn that this is a binary classifier only, see
        # https://scikit-learn.org/stable/developers/develop.html#estimator-tags
        return dict(binary_only=True)

    def __repr__(self, N_CHAR_MAX=700) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self._is_fitted():
            return 'Unfitted model'

        if self.root_ is None:
            return 'Empty model'

        return str(self.root_)
