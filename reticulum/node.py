from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np

from reticulum.util import multivariate_betaln, compute_log_p_data
from reticulum.util import sigmoid, d_log_multivariate_beta_d_alphas

# Adam optimizer settings (see https://arxiv.org/abs/1412.6980)
BETA_1_ADAM = 0.9
BETA_2_ADAM = 0.999
EPS_ADAM = 1e-8


# Note: The following three methods must adhere to the same node iteration order
# in order to avoid weight vectors getting wrongly assigned to nodes:
#
# - _collect_weight_matrix
# - _distribute_weight_matrix
# - _compute_d_err_d_s_hat_and_collect_d_err_d_weights


@dataclass
class Node:
    """
    A node in the adaptive Bayesian reticulum tree.
    """
    level: int

    left__child: Optional[Node] = None
    right_child: Optional[Node] = None

    weights: Optional[np.ndarray] = None

    s: Optional[np.ndarray] = None
    s_hat_left_: Optional[np.ndarray] = None
    s_hat_right: Optional[np.ndarray] = None

    k_left_: Optional[int] = None
    k_right: Optional[int] = None

    posterior_left_: Optional[np.ndarray] = None
    posterior_right: Optional[np.ndarray] = None

    log_p_data_left_: float = None
    log_p_data_right: float = None

    log_p_data_split = None
    log_p_data_no_split = None

    def try_fit(
            self,
            Xa: np.ndarray,
            y: np.ndarray,
            prior: np.ndarray,
            s_hat_parent: Optional[np.ndarray],
            learning_rate_init: float,
            n_gradient_descent_steps: Optional[int],
            initial_relative_stiffness: Optional[float]) -> bool:
        if s_hat_parent is None:
            n_data = Xa.shape[0]
            s_hat_parent = np.ones(n_data)

        # 1. initialize weights (if required)
        if self.weights is not None:
            # use weights that are already set on this node
            initial_weights = self.weights
        else:
            # initialize by choosing a random weight vector
            initial_weights = self._initialize_weights(
                Xa=Xa,
                s_hat_parent=s_hat_parent,
                initial_relative_stiffness=initial_relative_stiffness)

        if initial_weights is not None:
            self.weights = initial_weights
        else:
            # no data that we could split
            return False

        # 2. optimize weights of newly added node using gradient descent (i.e., local optimization)
        self._perform_gradient_descent(
            Xa=Xa,
            y=y,
            s_hat_parent=s_hat_parent,
            learning_rate_init=learning_rate_init,
            n_gradient_descent_steps=n_gradient_descent_steps,
            prior=prior)

        # 3. compute log-likelihoods of not splitting vs. splitting
        log_p_data_no_split = self._compute_log_p_data_no_split(y, s_hat_parent, prior)
        self.update_log_p_data_soft_split(Xa, y, s_hat_parent, prior, recursive=False)
        log_p_data_split = self.log_p_data_left_ + self.log_p_data_right

        # keep and complete
        self.posterior_left_ = prior + self.k_left_
        self.posterior_right = prior + self.k_right
        self.log_p_data_split = log_p_data_split
        self.log_p_data_no_split = log_p_data_no_split

        return True

    def try_prune(
            self,
            Xa: np.ndarray,
            y: np.ndarray,
            prior: np.ndarray,
            pruning_factor: float,
            verbose: bool,
            s_hat_parent: Optional[np.ndarray]=None) -> Tuple[bool, bool]:
        any_child_pruned = False

        # recursively try pruning children first because actual pruning must start at the leaf level
        if self.left__child is not None:
            left_child_pruned, any_child_of_left_child_pruned = self.left__child.try_prune(
                Xa=Xa,
                y=y,
                prior=prior,
                pruning_factor=pruning_factor,
                verbose=verbose,
                s_hat_parent=self.s_hat_left_)

            if left_child_pruned:
                self.left__child = None

            any_child_pruned |= left_child_pruned
            any_child_pruned |= any_child_of_left_child_pruned

        if self.right_child is not None:
            right_child_pruned, any_child_of_right_child_pruned = self.right_child.try_prune(
                Xa=Xa,
                y=y,
                prior=prior,
                pruning_factor=pruning_factor,
                verbose=verbose,
                s_hat_parent=self.s_hat_right)

            if right_child_pruned:
                self.right_child = None

            any_child_pruned |= right_child_pruned
            any_child_pruned |= any_child_of_right_child_pruned

        # now try pruning this node if it has no children (either because it never had or because they just got pruned)
        if self.left__child is None and self.right_child is None:
            if s_hat_parent is None:
                n_data = Xa.shape[0]
                s_hat_parent = np.ones(n_data)

            # check if this split is adding value
            pruning_factor_for_level = pruning_factor**(self.level+1)
            log_p_data_no_split = self._compute_log_p_data_no_split(y, s_hat_parent, prior)
            log_p_data_split = self.log_p_data_left_ + self.log_p_data_right
            if log_p_data_split <= log_p_data_no_split + np.log(pruning_factor_for_level):
                # this split isn't adding value (anymore) -> prune
                if verbose:
                    print(f'Pruning node at level {self.level}')

                return True, any_child_pruned

        # check if this split is actually splitting data
        activation = Xa @ self.weights
        if np.all(activation < 0) or np.all(activation > 0):
            # all data on one side -> not actually splitting anything
            return True, any_child_pruned

        return False, any_child_pruned

    def update_probability(self, Xa: np.ndarray, p: np.ndarray) -> None:
        if self.left__child is None:
            p += self.s_hat_left_.reshape(-1, 1) @ self.predict_proba_leaf_left_().reshape(1, -1)
        else:
            self.left__child.update_probability(Xa, p)

        if self.right_child is None:
            p += self.s_hat_right.reshape(-1, 1) @ self.predict_proba_leaf_right().reshape(1, -1)
        else:
            self.right_child.update_probability(Xa, p)

    def collect_terminal_nodes(self, is_left: bool, nodes: List[Node]=None) -> List[Node]:
        if nodes is None:
            nodes = []

        if self.left__child:
            self.left__child.collect_terminal_nodes(is_left, nodes)
        elif is_left:
            nodes.append(self)

        if self.right_child:
            self.right_child.collect_terminal_nodes(is_left, nodes)
        elif not is_left:
            nodes.append(self)

        return nodes

    @staticmethod
    def _initialize_weights(
            Xa: np.ndarray,
            s_hat_parent: np.ndarray,
            initial_relative_stiffness: float) -> Optional[np.ndarray]:
        assert s_hat_parent.ndim == 1

        n_dim = Xa.shape[1]-1  # ignore the augmented offset dimension

        # remove data points that have a too low weight
        keep_condition = s_hat_parent >= 0.5
        Xa = Xa[keep_condition, :]
        if len(Xa) < 2:
            # no point in splitting data sets with less than two points
            return None

        # compute outlier-resistant data range and compute weight scaling
        half_range = np.quantile(Xa[:, 1:], 0.75, axis=0) - np.quantile(Xa[:, 1:], 0.25, axis=0)
        zeros = half_range == 0
        if np.any(zeros):
            # first fix for zero entries: use half of the full range
            half_range[zeros] = (0.5 * (np.max(Xa[:, 1:], axis=0) - np.min(Xa[:, 1:], axis=0)))[zeros]

        zeros = half_range == 0
        if np.any(zeros):
            # second fix for zero entries: use 1
            half_range[zeros] = 1

        sd = 1/half_range

        # choose random initial weight appropriately scaled to the data
        weights = np.random.normal(0, 1, n_dim)
        weights /= np.linalg.norm(weights)
        weights *= sd

        # center the weight vector at the median of the data
        median = np.median(Xa[:, 1:], axis=0)
        weights = np.insert(weights, 0, 0)
        weights[0] = -np.dot(median, weights[1:])

        # apply initial stiffness
        return weights * initial_relative_stiffness

    def _perform_gradient_descent(
            self,
            Xa: np.ndarray,
            y: np.ndarray,
            s_hat_parent: np.ndarray,
            learning_rate_init: float,
            n_gradient_descent_steps: int,
            prior: np.ndarray) -> None:
        weight_matrix = self._collect_weight_matrix()
        momentum = 0
        velocity = 0

        if hasattr(self, 'callback') and hasattr(self, 'root'):
            self.callback('gd_start', self.root, self)

        for i in range(n_gradient_descent_steps):
            gradient = self._compute_d_err_d_weights(Xa=Xa, y=y, prior=prior, s_hat_parent=s_hat_parent)

            momentum = BETA_1_ADAM * momentum + (1 - BETA_1_ADAM) * gradient
            velocity = BETA_2_ADAM * velocity + (1 - BETA_2_ADAM) * gradient**2
            momentum_hat = momentum/(1 - BETA_1_ADAM**(i + 1))
            velocity_hat = velocity/(1 - BETA_2_ADAM**(i + 1))

            weight_matrix -= learning_rate_init * momentum_hat / (np.sqrt(velocity_hat) + EPS_ADAM)
            self._distribute_weight_matrix(weight_matrix)

        if hasattr(self, 'callback') and hasattr(self, 'root'):
            self.callback('gd_end', self.root, self)

    def update_s_hat(
            self, Xa: np.ndarray,
            s_hat_parent: Optional[np.ndarray]=None) -> None:
        if s_hat_parent is None:
            n_data = Xa.shape[0]
            s_hat_parent = np.ones(n_data)

        # compute left and right node outputs
        self.s = sigmoid(Xa @ self.weights)
        self.s_hat_left_ = s_hat_parent * self.s
        self.s_hat_right = s_hat_parent * (1-self.s)

        # recursively call children
        if self.left__child is not None:
            self.left__child.update_s_hat(Xa, self.s_hat_left_)

        if self.right_child is not None:
            self.right_child.update_s_hat(Xa, self.s_hat_right)

    def _compute_d_err_d_weights(
            self,
            Xa: np.ndarray,
            y: np.ndarray,
            prior: np.ndarray,
            s_hat_parent: np.ndarray) -> np.ndarray:
        # recursive forward pass
        self.update_s_hat(Xa, s_hat_parent)

        # recursive backward pass
        class_idx = [y == i for i in range(len(np.unique(y)))]
        d_k_d_s_hat = np.vstack([1*ci for ci in class_idx])

        d_err_d_weights_list = []
        self._compute_d_err_d_s_hat_and_collect_d_err_d_weights(
            Xa, prior, s_hat_parent, class_idx, d_k_d_s_hat, d_err_d_weights_list)

        return np.vstack(d_err_d_weights_list).T

    def _compute_d_err_d_s_hat_and_collect_d_err_d_weights(
            self,
            Xa: np.ndarray,
            prior: np.ndarray,
            s_hat_parent: np.ndarray,
            class_idx: List[np.ndarray],
            d_k_d_s_hat: np.ndarray,
            d_err_d_weights_list: List[np.ndarray]) -> np.ndarray:
        # see node iteration order note at the top of this file

        if self.left__child is None:
            # leaf on the left
            k_left_ = np.array([self.s_hat_left_[ci].sum() for ci in class_idx])
            posterior_left_ = prior + k_left_
            d_err_d_k_left_ = -d_log_multivariate_beta_d_alphas(posterior_left_)
            d_err_d_s_hat_left_ = d_err_d_k_left_ @ d_k_d_s_hat
        else:
            # child node on the left
            d_err_d_s_hat_left_ = self.left__child._compute_d_err_d_s_hat_and_collect_d_err_d_weights(
                Xa=Xa,
                prior=prior,
                s_hat_parent=self.s_hat_left_,
                class_idx=class_idx,
                d_k_d_s_hat=d_k_d_s_hat,
                d_err_d_weights_list=d_err_d_weights_list)

        if self.right_child is None:
            # leaf on the left
            k_right = np.array([self.s_hat_right[ci].sum() for ci in class_idx])
            posterior_right = prior + k_right
            d_err_d_k_right = -d_log_multivariate_beta_d_alphas(posterior_right)
            d_err_d_s_hat_right = d_err_d_k_right @ d_k_d_s_hat
        else:
            # child node on the left
            d_err_d_s_hat_right = self.right_child._compute_d_err_d_s_hat_and_collect_d_err_d_weights(
                Xa=Xa,
                prior=prior,
                s_hat_parent=self.s_hat_right,
                class_idx=class_idx,
                d_k_d_s_hat=d_k_d_s_hat,
                d_err_d_weights_list=d_err_d_weights_list)

        d_s_hat_left__ds = s_hat_parent
        d_s_hat_right_ds = -s_hat_parent
        d_err_d_s = d_err_d_s_hat_left_ * d_s_hat_left__ds + d_err_d_s_hat_right * d_s_hat_right_ds
        d_err_d_s_hat_parent = d_err_d_s_hat_left_ * self.s + d_err_d_s_hat_right * (1-self.s)
        d_s_d_a = self.s*(1-self.s)
        d_err_d_weights = (d_err_d_s * d_s_d_a) @ Xa
        d_err_d_weights_list.append(d_err_d_weights)

        return d_err_d_s_hat_parent

    def update_log_p_data_soft_split(
            self,
            Xa: np.ndarray,
            y: np.ndarray,
            s_hat_parent: np.ndarray,
            prior: np.ndarray,
            recursive: bool) -> None:
        assert Xa.shape[0] == len(y)
        assert s_hat_parent.ndim == 1

        class_idx = [y == i for i in range(len(np.unique(y)))]

        self.s = sigmoid(Xa @ self.weights)

        self.s_hat_left_ = s_hat_parent * self.s
        self.s_hat_right = s_hat_parent * (1 - self.s)

        self.k_left_ = Node._compute_k(self.s_hat_left_, class_idx).T
        self.k_right = Node._compute_k(self.s_hat_right, class_idx).T

        betaln_prior = multivariate_betaln(prior)
        self.log_p_data_left_ = compute_log_p_data(prior, self.k_left_, betaln_prior)
        self.log_p_data_right = compute_log_p_data(prior, self.k_right, betaln_prior)

        if recursive:
            if self.left__child:
                self.left__child.update_log_p_data_soft_split(Xa, y, self.s_hat_left_, prior, recursive)

            if self.right_child:
                self.right_child.update_log_p_data_soft_split(Xa, y, self.s_hat_right, prior, recursive)

    @staticmethod
    def _compute_log_p_data_no_split(
            y: np.ndarray,
            s_hat_parent: np.ndarray,
            prior: np.ndarray) -> np.ndarray:
        class_idx = [y == i for i in range(len(np.unique(y)))]
        k = Node._compute_k(s_hat_parent, class_idx).T
        betaln_prior = multivariate_betaln(prior)
        return compute_log_p_data(prior, k, betaln_prior)

    @staticmethod
    def _compute_k(s_hat: np.ndarray, class_idx: List[np.array]) -> np.ndarray:
        return np.array([s_hat[ci].sum(axis=0) for ci in class_idx])

    def _collect_weight_matrix(self, weights_list: List[np.ndarray]=None) -> Optional[np.ndarray]:
        # see node iteration order note at the top of this file

        if weights_list is None:
            weights_list = []
            return_matrix = True
        else:
            return_matrix = False

        if self.left__child is not None:
            self.left__child._collect_weight_matrix(weights_list)

        if self.right_child is not None:
            self.right_child._collect_weight_matrix(weights_list)

        weights_list.append(self.weights)

        if return_matrix:
            return np.array(weights_list).T

    def _distribute_weight_matrix(self, weight_matrix: np.ndarray, index: Optional[int]=None) -> int:
        # see node iteration order note at the top of this file

        if index is None:
            index = 0

        if self.left__child is not None:
            index = self.left__child._distribute_weight_matrix(weight_matrix, index)

        if self.right_child is not None:
            index = self.right_child._distribute_weight_matrix(weight_matrix, index)

        self.weights = weight_matrix[:, index]
        index += 1

        # now that the weights have changed the node is potentially splittable once again
        self.left__is_splittable = True
        self.right_is_splittable = True

        return index

    def update_depth(self, depth: int) -> int:
        depth = max(depth, self.level+1)

        if self.left__child is not None:
            depth = self.left__child.update_depth(depth)

        if self.right_child is not None:
            depth = self.right_child.update_depth(depth)

        return depth

    def update_n_leaves(self, n_leaves: int) -> int:
        if self.left__child is None:
            n_leaves += 1
        else:
            n_leaves = self.left__child.update_n_leaves(n_leaves)

        if self.right_child is None:
            n_leaves += 1
        else:
            n_leaves = self.right_child.update_n_leaves(n_leaves)

        return n_leaves

    def update_feature_importance(self, feature_importance: np.ndarray) -> None:
        # the more the normal vector is oriented along a given dimension's axis the
        # more important that dimension is, so weight the gain in log-likelihood by
        # the absolute value of the unit hyperplane normal
        log_p_gain = self.log_p_data_split - self.log_p_data_no_split
        hyperplane_normal = self.weights[1:] / np.linalg.norm(self.weights[1:])
        feature_importance += log_p_gain * np.abs(hyperplane_normal)

        if self.left__child is not None:
            self.left__child.update_feature_importance(feature_importance)

        if self.right_child is not None:
            self.right_child.update_feature_importance(feature_importance)

    def predict_proba_leaf_left_(self) -> np.ndarray:
        if self.posterior_left_ is None:
            return np.nan

        return self.posterior_left_ / self.posterior_left_.sum()

    def predict_proba_leaf_right(self) -> np.ndarray:
        if self.posterior_right is None:
            return np.nan

        return self.posterior_right / self.posterior_right.sum()

    def get_n_data(self) -> float:
        return self.k_left_.sum() + self.k_right.sum()

    def __str__(self) -> str:
        return self._str('', '')

    def _str(self, prefix: str, children_prefix: str) -> str:
        s = prefix

        # choose hyperplane origin to be the closest point to the origin
        weights = self.weights
        normal = weights[1:]
        origin = normal * -weights[0] / np.dot(normal, normal)
        origin_str = np.array2string(origin, max_line_width=9999, separator=', ', floatmode='maxprec_equal')
        normal_str = np.array2string(normal, max_line_width=9999, separator=', ', floatmode='maxprec_equal')
        s += f'origin={origin_str}, normal={normal_str}'
        s += '\n'

        child_prefix_left_ = children_prefix + '├─ left: '
        child_prefix_right = children_prefix + '└─ right:'
        if self.left__child is not None:
            s += self.left__child._str(child_prefix_left_, children_prefix + '│  ')
        else:
            s += f'{child_prefix_left_} p(y)={self.predict_proba_leaf_left_()}, n={self.k_left_.sum()}\n'

        if self.right_child is not None:
            s += self.right_child._str(child_prefix_right, children_prefix + '   ')
        else:
            s += f'{child_prefix_right} p(y)={self.predict_proba_leaf_right()}, n={self.k_right.sum()}\n'

        return s
