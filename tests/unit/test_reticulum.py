from unittest import TestCase

import numpy as np
import pandas as pd
from numpy.random import normal, randint
from numpy.testing import assert_array_equal, assert_array_almost_equal
from sklearn.utils.estimator_checks import check_estimator

from reticulum import AdaptiveBayesianReticulum

# possible data matrix types/transforms that need to work for fit()
data_matrix_transforms = [
    lambda X: X,
    lambda X: pd.DataFrame(data=X, columns=[f'col-{i}' for i in range(len(X[0]))]),
]


class AdaptiveBayesianReticulumTest(TestCase):
    def test_sklearn_compatible_estimator(self) -> None:
        check_estimator(AdaptiveBayesianReticulum)
        check_estimator(AdaptiveBayesianReticulum())

    def test_cannot_fit_with_bad_dimensions(self) -> None:
        np.random.seed(6666)

        model = AdaptiveBayesianReticulum(prior=(1, 1), learning_rate_init=1e-3, random_state=666)
        for good_X in [normal(0, 1, [10, 10])]:
            for bad_y in [randint(0, 2, []), randint(0, 2, [10, 10]), randint(0, 2, [11]), randint(0, 2, [10, 10, 10])]:
                try:
                    model.fit(good_X, bad_y)
                    self.fail()
                except ValueError:
                    pass

        for bad_X in [normal(0, 1, [10, 10, 10])]:
            for good_y in [randint(0, 2, [10])]:
                try:
                    model.fit(bad_X, good_y)
                    self.fail()
                except ValueError:
                    pass

    def test_cannot_predict_before_training(self) -> None:
        model = AdaptiveBayesianReticulum(random_state=666)

        # can't predict yet
        try:
            model.predict([])
            self.fail()
        except ValueError:
            pass

        # can't predict probability yet
        try:
            model.predict_proba([])
            self.fail()
        except ValueError:
            pass

    def test_cannot_predict_with_bad_input_dimensions(self) -> None:
        for data_matrix_transform in data_matrix_transforms:
            model = AdaptiveBayesianReticulum(prior=(1, 1), learning_rate_init=1e-3, random_state=666)

            Xy = np.array([
                [0.0, 0, 0],
                [0.1, 1, 0],

                [0.9, 0, 1],
                [1.0, 1, 1],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            print(f'Testing {type(X).__name__}')
            model.fit(X, y)
            print(model)

            model.predict([[0, 0]])

            try:
                model.predict(0)
                self.fail()
            except ValueError:
                pass

            try:
                model.predict([[0]])
                self.fail()
            except ValueError:
                pass

            try:
                model.predict([[0, 0, 0]])
                self.fail()
            except ValueError:
                pass

    def test_print_empty_model(self) -> None:
        model = AdaptiveBayesianReticulum(random_state=666)
        print(model)

    def test_no_split_1(self) -> None:
        for data_matrix_transform in data_matrix_transforms:
            model = AdaptiveBayesianReticulum(prior=(1, 1), pruning_factor=2, random_state=666)

            Xy = np.array([
                [0.0, 0, 0],
                [0.0, 1, 1],
                [1.0, 2, 0],
                [1.0, 3, 1],
                [1.0, 4, 0],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            print(f'Testing {type(X).__name__}')
            model.fit(X, y)
            print(model)

            self.assertEqual(model.get_depth(), 0)
            self.assertEqual(model.get_n_leaves(), 0)
            self.assertTrue(model._is_fitted())
            self.assertIsNone(model.root_)

            self.assertEqual(model.predict([[0, 0]]), np.zeros(1))
            assert_array_equal(model.predict_proba([[0, 0], [11, 99]]), [[0.6, 0.4], [0.6, 0.4]])

    def test_no_split_2(self) -> None:
        for data_matrix_transform in data_matrix_transforms:
            model = AdaptiveBayesianReticulum(prior=(1, 1), pruning_factor=2, random_state=666)

            Xy = np.array([
                [0.0, 0, 1],
                [0.0, 1, 0],
                [1.0, 2, 1],
                [1.0, 3, 0],
                [1.0, 4, 1],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            print(f'Testing {type(X).__name__}')
            model.fit(X, y)
            print(model)

            self.assertEqual(model.get_depth(), 0)
            self.assertEqual(model.get_n_leaves(), 0)
            self.assertTrue(model._is_fitted())
            self.assertIsNone(model.root_)

            self.assertEqual(model.predict([[0, 0]]), np.ones(1))
            assert_array_equal(model.predict_proba([[0, 0], [11, 99]]), [[0.4, 0.6], [0.4, 0.6]])

    def test_one_split(self) -> None:
        for data_matrix_transform in data_matrix_transforms:
            model = AdaptiveBayesianReticulum(
                prior=(1, 1),
                learning_rate_init=5e-2,
                initial_relative_stiffness=2,
                random_state=666)

            Xy = np.array([
                [0.0, 0, 0],
                [0.1, 1, 0],

                [0.9, 0, 1],
                [1.0, 1, 1],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            print(f'Testing {type(X).__name__}')
            model.fit(X, y)
            print(model)

            self.assertEqual(model.get_depth(), 1)
            self.assertEqual(model.get_n_leaves(), 2)

            self.assertIsNone(model.root_.left__child)
            self.assertIsNone(model.root_.right_child)

            x_axis_intersection = -model.root_.weights[0] / model.root_.weights[1]
            normal_slope = model.root_.weights[2] / model.root_.weights[1]
            self.assertTrue(0.4 < x_axis_intersection < 0.6)
            self.assertTrue(-0.15 < normal_slope < 0.15)

            expected = np.array([0, 0, 1, 1])
            self.assertEqual(model.predict([[0, 0]]), expected[0])
            self.assertEqual(model.predict([[0, 1]]), expected[1])
            self.assertEqual(model.predict([[1, 0]]), expected[2])
            self.assertEqual(model.predict([[1, 1]]), expected[3])

            for data_matrix_transform2 in data_matrix_transforms:
                assert_array_equal(model.predict(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 0]])), expected)

            expected = np.array([[3/4, 1/4], [3/4, 1/4], [1/4, 3/4], [1/4, 3/4]])
            assert_array_almost_equal(model.predict_proba([[0, 0]]), np.expand_dims(expected[0], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[0, 1]]), np.expand_dims(expected[1], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[1, 0]]), np.expand_dims(expected[2], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[1, 1]]), np.expand_dims(expected[3], 0), decimal=1)

            for data_matrix_transform2 in data_matrix_transforms:
                assert_array_almost_equal(model.predict_proba(data_matrix_transform2([[0, 0], [0, 1], [1, 0], [1, 0]])),
                                          expected, decimal=1)

    def test_two_splits(self) -> None:
        for data_matrix_transform in data_matrix_transforms:
            model = AdaptiveBayesianReticulum(
                prior=(1, 1),
                learning_rate_init=1e-1,
                n_gradient_descent_steps=1000,
                initial_relative_stiffness=2,
                random_state=666)

            Xy = np.array([
                [0.0, 0.0, 0],
                [0.0, 0.3, 0],
                [0.0, 0.7, 0],
                [0.0, 1.0, 0],

                [1.0, 0.1, 1],
                [1.0, 0.2, 1],
                [1.0, 0.8, 1],
                [1.0, 0.9, 1],

                [2.0, 0.4, 0],
                [2.0, 0.6, 0],
            ])
            X = Xy[:, :-1]
            y = Xy[:, -1]

            X = data_matrix_transform(X)

            print(f'Testing {type(X).__name__}')
            model.fit(X, y)
            print(model)

            self.assertEqual(model.get_depth(), 2)
            self.assertEqual(model.get_n_leaves(), 3)
            self.assertIsNone(model.root_.right_child)
            self.assertIsNotNone(model.root_.left__child)

            x_axis_intersection_0 = -model.root_.weights[0] / model.root_.weights[1]
            x_axis_intersection_1 = -model.root_.left__child.weights[0] / model.root_.left__child.weights[1]
            y_axis_intersection_0 = -model.root_.weights[0] / model.root_.weights[2]
            y_axis_intersection_1 = -model.root_.left__child.weights[0] / model.root_.left__child.weights[2]
            self.assertTrue(0.4 < x_axis_intersection_0 < 0.6, 'expected 1st split to cross x-axis around 0.5')
            self.assertTrue(1.4 < x_axis_intersection_1 < 1.6, 'expected 2nd split to cross x-axis around 1.5')
            self.assertTrue(abs(y_axis_intersection_0) > 15, 'expected 1st split to cross y-axis far away from 0')
            self.assertTrue(abs(y_axis_intersection_1) > 15, 'expected 2nd split to cross y-axis far away from 0')

            expected = np.array([0, 0, 1, 1, 0, 0])
            self.assertEqual(model.predict([[0, 0.5]]), expected[0])
            self.assertEqual(model.predict([[0.4, 0.5]]), expected[1])
            self.assertEqual(model.predict([[0.6, 0.5]]), expected[2])
            self.assertEqual(model.predict([[1.4, 0.5]]), expected[3])
            self.assertEqual(model.predict([[1.6, 0.5]]), expected[4])
            self.assertEqual(model.predict([[100, 0.5]]), expected[5])

            for data_matrix_transform2 in data_matrix_transforms:
                assert_array_equal(model.predict(data_matrix_transform2(
                    [[0.0, 0.5], [0.4, 0.5], [0.6, 0.5], [1.4, 0.5], [1.6, 0.5], [100, 0.5]])
                ), expected)

            expected = np.array([[5/6, 1/6], [5/6, 1/6], [1/6, 5/6], [1/6, 5/6], [3/4, 1/4], [3/4, 1/4]])
            assert_array_almost_equal(model.predict_proba([[0, 0.5]]), np.expand_dims(expected[0], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[0.4, 0.5]]), np.expand_dims(expected[1], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[0.6, 0.5]]), np.expand_dims(expected[2], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[1.4, 0.5]]), np.expand_dims(expected[3], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[1.6, 0.5]]), np.expand_dims(expected[4], 0), decimal=1)
            assert_array_almost_equal(model.predict_proba([[100, 0.5]]), np.expand_dims(expected[5], 0), decimal=1)

            for data_matrix_transform2 in data_matrix_transforms:
                assert_array_almost_equal(model.predict_proba(data_matrix_transform2(
                    [[0.0, 0.5], [0.4, 0.5], [0.6, 0.5], [1.4, 0.5], [1.6, 0.5], [100, 0.5]])
                ), expected, decimal=1)

    def test_compute_d_err_d_weights(self) -> None:
        # create a toy problem
        prior = (1, 1)
        model = AdaptiveBayesianReticulum(
            prior=prior,
            n_gradient_descent_steps=0,
            initial_relative_stiffness=10,
            random_state=1)

        n_data = 1000
        n_dim = 2

        np.random.seed(666)
        X = np.vstack((
            np.random.normal(3, 1e-1, (n_data//2, n_dim)),
            np.random.normal(4, 1e-1, (n_data//2, n_dim))
        ))
        y = np.hstack((np.zeros(n_data//2), np.ones(n_data//2)))

        # 'fit' the model by just performing initial weight search but not doing any gradient descent
        model.fit(X, y)
        self.assertGreaterEqual(model.get_n_leaves(), 3, 'We expect a few nodes to emerge from this toy problem')

        # compute derivative analytically
        s_hat_parent = np.ones(n_data)
        Xa = np.hstack((s_hat_parent.reshape(-1, 1), X))
        d_err_d_weights_analytic = model.root_._compute_d_err_d_weights(Xa, y, np.asarray(prior), s_hat_parent)

        # compute derivative numerically
        d_err_d_weights_numeric = np.zeros(d_err_d_weights_analytic.shape)
        dw = 1e-6
        terminal_nodes_left_ = model.root_.collect_terminal_nodes(is_left=True)
        terminal_nodes_right = model.root_.collect_terminal_nodes(is_left=False)
        weight_matrix_bak = model.root_._collect_weight_matrix()
        for i in range(weight_matrix_bak.shape[0]):
            for j in range(weight_matrix_bak.shape[1]):
                # compute 2nd order numerical approximation of derivative
                weight_matrix = weight_matrix_bak.copy()
                weight_matrix[i, j] += dw
                model.root_._distribute_weight_matrix(weight_matrix)
                model.root_.update_s_hat(Xa, s_hat_parent)
                model.root_.update_log_p_data_soft_split(Xa, y, s_hat_parent, prior, recursive=True)
                err1 = -np.sum([node.log_p_data_left_ for node in terminal_nodes_left_]) \
                       -np.sum([node.log_p_data_right for node in terminal_nodes_right])

                weight_matrix = weight_matrix_bak.copy()
                weight_matrix[i, j] -= dw
                model.root_._distribute_weight_matrix(weight_matrix)
                model.root_.update_log_p_data_soft_split(Xa, y, s_hat_parent, prior, recursive=True)
                err2 = -np.sum([node.log_p_data_left_ for node in terminal_nodes_left_]) \
                       -np.sum([node.log_p_data_right for node in terminal_nodes_right])

                d_err_d_weights_numeric[i, j] = (err1-err2)/(2*dw)

        assert_array_almost_equal(d_err_d_weights_analytic, d_err_d_weights_numeric)
