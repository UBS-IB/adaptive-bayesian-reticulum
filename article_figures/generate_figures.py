# Generates all charts of the paper.
#
# If you are behind a corporate proxy you need to specify the proxies by supplying
# the following environment variables:
# - HTTP_PROXY in the form:  http://user:password@your-http-proxy:port
# - HTTPS_PROXY in the form: https://user:password@your-https-proxy:port

import os
from typing import Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from reticulum import AdaptiveBayesianReticulum
from examples.util import load_ripley

seed = 42
n_jobs = 5
font_size = 14
title_size = 20
figure_suffixes = ['.pdf']
show_plot = True
print_metrics = True


def load_csv(file_name: str) -> Tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(file_name).to_numpy()
    return data[:, :-1], data[:, -1]


def plot_model(
        model_name: str,
        model: Any,
        file_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        c0: np.ndarray,
        c1: np.ndarray,
        elevation: float,
        marker_size: int,
        alpha: float) -> None:
    # grid
    x_grid = np.linspace(x_lim[0], x_lim[1], 301)
    y_grid = np.linspace(y_lim[0], y_lim[1], 301)
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)
    xy = np.vstack((x_grid.ravel(), y_grid.ravel())).T
    grid_predictions = model.predict_proba(xy)[:, 1].reshape(x_grid.shape)

    # plot
    cmap = plt.get_cmap('coolwarm')
    plot_3d(file_name + '_surf', x_grid, y_grid, grid_predictions, elevation, alpha)
    plot_2d(
        file_name + '_cont',
        model_name,
        x_grid,
        y_grid,
        grid_predictions,
        x_lim,
        y_lim,
        c0,
        c1,
        marker_size,
        cmap)

    if print_metrics:
        # compute and print in-sample vs. 5-fold CV score for sanity checks
        accuracy = accuracy_score(y_train, model.predict(X_train))
        print(f'{model_name}: in-sample accuracy: {100*accuracy:.1f} %')

        for scoring in ['accuracy', 'neg_log_loss']:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            np.random.seed(seed)
            score = cross_val_score(model, X_train, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs)
            print(f'{model_name}: 5-fold CV {scoring}: {score.mean()}')

        # print node counts
        if isinstance(model, AdaptiveBayesianReticulum):
            # print model stats
            print(f'Depth:  {model.get_depth()}')
            print(f'Leaves: {model.get_n_leaves()}')
            print(f'Node count: {model.get_n_leaves() - 1}')
            print(model)

        if isinstance(model, DecisionTreeClassifier):
            print(f'Node count: {model.tree_.node_count - model.tree_.n_leaves}')

        if isinstance(model, RandomForestClassifier):
            internal_nodes_count = 0
            for estimator in model.estimators_:
                internal_nodes_count += estimator.tree_.node_count - estimator.tree_.n_leaves

            print(f'Node count: {internal_nodes_count}')

        if isinstance(model, XGBClassifier):
            df = model.get_booster().trees_to_dataframe()
            node_count = len(df[df.Feature != 'Leaf'])
            print(f'Node count: {node_count}')

        if isinstance(model, MLPClassifier):
            node_count = sum(model.get_params()['hidden_layer_sizes'])
            print(f'Node count: {node_count + 1}')


def plot_3d(
        file_name: str,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        predictions: np.ndarray,
        elevation: float,
        alpha: float):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(
        x_grid,
        y_grid,
        predictions,
        rstride=1,
        cstride=1,
        color='silver',
        shade=True,
        linewidth=1,
        antialiased=False,  # avoids artifacts
        alpha=alpha,
        vmin=0,
        vmax=1)

    ax.set_xlabel('$x_1$', fontsize=font_size)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    ax.set_ylabel('$x_2$', fontsize=font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    ax.set_zlabel('$p$', fontsize=font_size)
    for tick in ax.zaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    ax.view_init(elevation, 180+70)
    plt.tick_params(axis='both', which='major')
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_zlim(0.0, 1.0)
    ax.zaxis.set_major_locator(plt.MultipleLocator(0.2))
    ax.set_zticklabels(np.round(np.arange(0.0, 1.2, .2) * 10.0) / 10.0)
    ax.autoscale_view(tight=True)
    for figure_suffix in figure_suffixes:
        fig.savefig(file_name + figure_suffix, dpi=600)

    if show_plot:
        plt.show()


def plot_2d(
        file_name: str,
        title: str,
        x_grid: np.ndarray,
        y_grid: np.ndarray,
        predictions: np.ndarray,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
        c0: np.ndarray,
        c1: np.ndarray,
        size,
        cmap: Colormap) -> None:
    fig = plt.figure()
    ax = plt.gca()

    levels = np.linspace(0, 1, 11)
    cntr1 = ax.contourf(x_grid, y_grid, predictions, levels, alpha=0.85, cmap=cmap, zorder=1)

    ax.scatter(c0[:, 0], c0[:, 1], c=cmap(0.0), s=size, zorder=2)
    ax.scatter(c1[:, 0], c1[:, 1], c=cmap(1.0), s=size, zorder=3)

    ax.contour(x_grid, y_grid, predictions, [0.5], linewidths=1, linestyles='dashed', zorder=40000000)

    ax.set_xlabel('$x_1$', fontsize=font_size)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    ax.set_ylabel('$x_2$', fontsize=font_size)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(font_size)

    cbar = fig.colorbar(cntr1, ax=ax, fraction=0.03)
    cbar.set_label('$p$')
    cbar.ax.tick_params(labelsize=font_size)
    # ax.plot(df_input[x].values, df_input[y].values, 'ko', ms=3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_title(title, fontsize=title_size, y=1.04)
    plt.xticks(rotation='vertical')
    ax.xaxis.label.set_visible(False)
    ax.yaxis.label.set_visible(False)
    ax.set_aspect(1)
    plt.tight_layout()
    for figure_suffix in figure_suffixes:
        fig.savefig(file_name + figure_suffix, dpi=600)

    if show_plot:
        plt.show()


if __name__ == '__main__':
    # read proxies from environment variables (if required, see comment at the top of this file)
    proxies = {
        'http': os.environ.get('HTTP_PROXY', None),
        'https': os.environ.get('HTTPS_PROXY', None)
    }

    reticulum_title = 'Adaptive Bayesian Reticulum'
    rf_title = 'Random Forest'
    xgb_title = 'XGBoost'
    dt_title = 'Decision Tree'
    nn_title = 'Neural Network'

    # Ripley dataset
    print()
    print('Ripley dataset')
    print('--------------')
    train, _ = load_ripley(proxies)
    X_train = train[:, :-1]
    y_train = train[:, -1]
    x_lim = (-1.25, 1)
    y_lim = (-0.20, 1.2)
    c0 = X_train[y_train < 0.5]
    c1 = X_train[y_train >= 0.5]

    elevation = 45
    marker_size = 7
    alpha = 1

    if 1:
        params = {
            'n_pseudo_obs': 1.1982176593167126,
            'n_iter': 35,
            'learning_rate_init': 0.06823423186986995,
            'n_gradient_descent_steps': 93,
            'initial_relative_stiffness': 1.8960788813445533,
            'pruning_factor': 1.0237544874453033
        }

        model = AdaptiveBayesianReticulum(
            prior=(params['n_pseudo_obs'], params['n_pseudo_obs']),
            pruning_factor=params['pruning_factor'],
            n_iter=params['n_iter'],
            learning_rate_init=params['learning_rate_init'],
            n_gradient_descent_steps=params['n_gradient_descent_steps'],
            initial_relative_stiffness=params['initial_relative_stiffness'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(reticulum_title, model, 'ripley_ret', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 0:
        params = {
            'max_depth': 3.8728996919039727,
            'learning_rate_init': 0.45889620975631407,
            'booster': 'gbtree',
            'n_estimators': 10.129853666995231,
            'subsample': 0.7310565107306015,
            'colsample_bytree': 0.8383950235711187,
            'colsample_bylevel': 0.9837953284629628,
            'colsample_bynode': 0.9282471720388992
        }

        model = XGBClassifier(
            max_depth=int(params['max_depth']),
            learning_rate_init=params['learning_rate_init'],
            booster=params['booster'],
            n_estimators=int(params['n_estimators']),
            subsample=params['subsample'],
            colsample_bytree=params['colsample_bytree'],
            colsample_bylevel=params['colsample_bylevel'],
            colsample_bynode=params['colsample_bynode'],
            random_state=666)

        model.fit(X_train, y_train)
        plot_model(xgb_title, model, 'ripley_xgb', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 0:
        params = {
            'n_estimators': 15,
            'min_samples_leaf': 14,
            'max_depth': 25,
            'criterion': 'entropy'
        }

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(rf_title, model, 'ripley_rf', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 0:
        params = {
            'min_samples_leaf': 20,
            'max_depth': 19,
            'criterion': 'gini'
        }

        model = DecisionTreeClassifier(
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(dt_title, model, 'ripley_dt', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 0:
        params = {
            'n_hidden_nodes': 23,
            'activation': 'relu',
            'learning_rate_init': 0.023851334283313347
        }

        model = MLPClassifier(
            hidden_layer_sizes=(params['n_hidden_nodes'],),
            activation=params['activation'],
            solver='adam',
            learning_rate_init=params['learning_rate_init'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(nn_title, model, 'ripley_nn', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    # Circle dataset
    print()
    print('Circle dataset')
    print('--------------')
    X_train, y_train = load_csv('circle_train.csv')
    x_lim = (-2.0, 2.0)
    y_lim = (-2.0, 2.0)
    c0 = X_train[y_train < 0.5]
    c1 = X_train[y_train >= 0.5]

    elevation = 35
    marker_size = 2
    alpha = 1

    if 1:
        params = {
            'n_pseudo_obs': 1.567512774761449,
            'n_iter': 86,
            'learning_rate_init': 0.5459049825557916,
            'n_gradient_descent_steps': 100,
            'initial_relative_stiffness': 9.787487528997882,
            'pruning_factor': 1.0633657210992458
        }

        model = AdaptiveBayesianReticulum(
            prior=(params['n_pseudo_obs'], params['n_pseudo_obs']),
            pruning_factor=params['pruning_factor'],
            n_iter=params['n_iter'],
            learning_rate_init=params['learning_rate_init'],
            n_gradient_descent_steps=params['n_gradient_descent_steps'],
            initial_relative_stiffness=params['initial_relative_stiffness'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(reticulum_title, model, 'circle_ret', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 0:
        params = {
            'n_estimators': 446,
            'min_samples_leaf': 17,
            'max_depth': 20,
            'criterion': 'entropy'
        }

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(rf_title, model, 'circle_rf', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 1:
        params = {
            'max_depth': 4.926356749545193,
            'learning_rate_init': 0.05526210460085713,
            'n_estimators': 125.4858770588833,
            'subsample': 0.6010251431741602
        }

        model = XGBClassifier(
            max_depth=int(params['max_depth']),
            learning_rate_init=params['learning_rate_init'],
            n_estimators=int(params['n_estimators']),
            subsample=params['subsample'],
            random_state=666)

        model.fit(X_train, y_train)
        plot_model(xgb_title, model, 'circle_xgb', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 1:
        params = {
            'min_samples_leaf': 20,
            'max_depth': 4,
            'criterion': 'gini'
        }

        model = DecisionTreeClassifier(
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(dt_title, model, 'circle_dt', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 1:
        params = {
            'n_hidden_nodes': 22,
            'activation': 'relu',
            'learning_rate_init': 0.004731872586911062
        }

        model = MLPClassifier(
            hidden_layer_sizes=(params['n_hidden_nodes'],),
            activation=params['activation'],
            solver='adam',
            learning_rate_init=params['learning_rate_init'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(nn_title, model, 'circle_nn', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    # Cross dataset
    print()
    print('Cross dataset')
    print('-------------')
    X_train, y_train = load_csv('cross_train.csv')
    x_lim = (-2.0, 2.0)
    y_lim = (-2.0, 2.0)
    c0 = X_train[y_train < 0.5]
    c1 = X_train[y_train >= 0.5]

    elevation = 35
    marker_size = 5
    alpha = 0.8

    if 1:
        params = {
            'n_pseudo_obs': 1.000329117076533,
            'n_iter': 10,
            'learning_rate_init': 0.42573632846161397,
            'n_gradient_descent_steps': 172,
            'initial_relative_stiffness': 1.1614575976863115,
            'pruning_factor': 1.0895876478584843
        }

        model = AdaptiveBayesianReticulum(
            prior=(params['n_pseudo_obs'], params['n_pseudo_obs']),
            pruning_factor=params['pruning_factor'],
            n_iter=params['n_iter'],
            learning_rate_init=params['learning_rate_init'],
            n_gradient_descent_steps=params['n_gradient_descent_steps'],
            initial_relative_stiffness=params['initial_relative_stiffness'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(reticulum_title, model, 'cross_ret', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 0:
        params = {
            'n_estimators': 64,
            'min_samples_leaf': 6,
            'max_depth': 14,
            'criterion': 'gini'
        }

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(rf_title, model, 'cross_rf', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 1:
        params = {
            'max_depth': 3.3234011905573526,
            'learning_rate_init': 0.06284647216488741,
            'n_estimators': 109.40596980557514,
            'subsample': 0.6355910496057015
        }

        model = XGBClassifier(
            max_depth=int(params['max_depth']),
            learning_rate_init=params['learning_rate_init'],
            n_estimators=int(params['n_estimators']),
            subsample=params['subsample'],
            random_state=666)

        model.fit(X_train, y_train)
        plot_model(xgb_title, model, 'cross_xgb', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 1:
        params = {
            'min_samples_leaf': 10,
            'max_depth': 4,
            'criterion': 'gini'
        }

        model = DecisionTreeClassifier(
            criterion=params['criterion'],
            max_depth=params['max_depth'],
            min_samples_leaf=params['min_samples_leaf'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(dt_title, model, 'cross_dt', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)

    if 1:
        params = {
            'n_hidden_nodes': 25,
            'activation': 'relu',
            'learning_rate_init': 0.01184452220748257
        }

        model = MLPClassifier(
            hidden_layer_sizes=(params['n_hidden_nodes'],),
            activation=params['activation'],
            solver='adam',
            learning_rate_init=params['learning_rate_init'],
            random_state=seed)

        model.fit(X_train, y_train)
        plot_model(nn_title, model, 'cross_nn', X_train, y_train, x_lim, y_lim, c0, c1, elevation, marker_size, alpha)
