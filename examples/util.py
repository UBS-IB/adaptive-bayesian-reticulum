"""
A collection of publicly available data sets to test classification models on,
plus some helper functions for plotting.
"""

import io
from typing import List, Dict, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import Colormap
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer


def load_credit(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    content = requests.get(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls',
        proxies=proxies).content
    df = pd.read_excel(io.BytesIO(content))
    train = df.iloc[1:, 1:].values.astype(np.float64)
    train = _one_hot_encode(train, [2, 3])  # one-hot encode categorical features
    test = train
    return train, test


def load_dermatology(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # Dermatology
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data', proxies)
    lines = text.split('\n')[:-1]
    lines = [line for line in lines if '?' not in line]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    train[:, -1] -= 1
    test = train
    return train, test


def load_diabetic(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # Diabetic Retinopathy
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff', proxies)
    text = text[text.index('@data'):]
    lines = text.split('\n')[1:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    test = train
    return train, test


def load_eeg(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load EEG eye data
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff', proxies)
    text = text[text.index('@DATA'):]
    lines = text.split('\n')[1:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    test = train
    return train, test


def load_gamma(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load MAGIC Gamma telescope data
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data', proxies)
    text = text.replace('g', '0').replace('h', '1')
    lines = text.split('\n')[:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    test = train
    return train, test


def load_glass(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load glass identificaion data
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', proxies)
    lines = text.split('\n')[:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    train = train[:, 1:]  # ignore ID row
    train[:, -1] -= 1  # convert 1..7 to 0..6
    train[np.where(train[:, -1] >= 4)[0], -1] -= 1  # skip missing class
    test = train
    return train, test


def load_haberman(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load Haberman's dataset
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', proxies)
    lines = text.split('\n')[:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    train[:, -1] -= 1
    test = train
    return train, test


def load_heart(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # heart disease dataset
    text = _scrape(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat', proxies)
    lines = text.split('\n')[:-1]
    train = np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines))])
    train = _one_hot_encode(train, [2, 6, 12])  # one-hot encode categorical features
    train[:, -1] -= 1
    test = train
    return train, test


def load_iris(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # iris flower dataset
    text = _scrape(
        'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', proxies)
    lines = text.split('\n')[:-2]
    train = np.vstack([lines[i].split(',') for i in range(len(lines))])
    X = train[:, :-1].astype(np.float64)
    _, y = np.unique(train[:, -1], return_inverse=True)
    train = np.hstack((X, y.reshape(-1, 1)))
    test = train
    return train, test


def load_ripley(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load Ripley's synthetic dataset
    def parse_ripley(text: str) -> np.ndarray:
        lines = text.split('\n')[1:]
        return np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines)-1)])

    train = parse_ripley(_scrape('https://www.stats.ox.ac.uk/pub/PRNN/synth.tr', proxies))
    test = parse_ripley(_scrape('https://www.stats.ox.ac.uk/pub/PRNN/synth.te', proxies))
    return train, test


def load_seeds(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load wheat seeds dataset
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt', proxies)
    lines = text.split('\n')[:-1]
    train = np.vstack([np.fromstring(lines[i], sep=' ') for i in range(len(lines))])
    train[:, -1] -= 1
    test = train
    return train, test


def load_seismic(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load seismic bumps dataset
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff', proxies)
    text = text[text.index('@data'):]
    text = text.replace('a', '0').replace('b', '1').replace('c', '2').replace('d', '3')
    text = text.replace('N', '0').replace('W', '1')
    lines = text.split('\n')[1:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    test = train
    return train, test


def load_spambase(proxies: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    # load spam data
    text = _scrape('https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data', proxies)
    lines = text.split('\n')[:-1]
    train = np.vstack([np.fromstring(lines[i], sep=',') for i in range(len(lines))])
    test = train
    return train, test


def plot_2d_hyperplane(
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        info_train: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        info_test: str) -> None:
    plt.figure(figsize=[10, 16], dpi=75)

    n_classes = int(y_train.max()) + 1
    cmap = plt.get_cmap('coolwarm')

    x_min = min(X_train[:, 0].min(), X_test[:, 0].min())
    x_max = max(X_train[:, 0].max(), X_test[:, 0].max())
    y_min = min(X_train[:, 1].min(), X_test[:, 1].min())
    y_max = max(X_train[:, 1].max(), X_test[:, 1].max())

    x_lim = [x_min-0.2*(x_max-x_min), x_max+0.2*(x_max-x_min)]
    y_lim = [y_min-0.2*(y_max-y_min), y_max+0.2*(y_max-y_min)]

    def plot(X: np.ndarray, y: np.ndarray, info: str, cmap: Colormap) -> None:
        for i in range(n_classes):
            class_i = y == i
            plt.plot(X[np.where(class_i)[0], 0],
                     X[np.where(class_i)[0], 1],
                     'o',
                     ms=4,
                     c=cmap(i/(n_classes-1)),
                     label=f'Class {i}')

        draw_node_2d_hyperplane(model, x_lim, y_lim, cmap)

        plt.title(info)
        plt.xlabel('x0')
        plt.ylabel('x1')
        plt.legend()

    plt.subplot(211)
    plot(X_train, y_train, info_train, cmap)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect(1)

    plt.subplot(212)
    plot(X_test, y_test, info_test, cmap)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.gca().set_aspect(1)

    plt.show()


def draw_node_2d_hyperplane(model: Any, x_lim: List[float], y_lim: List[float], cmap: Colormap) -> None:
    n = 100
    x = np.linspace(x_lim[0], x_lim[1], n)
    y = np.linspace(y_lim[0], y_lim[1], n)
    xg, yg = np.meshgrid(x, y)
    X = np.array([xg.flatten(), yg.flatten()]).T
    p = model.predict_proba(X)[:, 1]
    p = p.reshape(xg.shape)

    levels = np.linspace(0, 1, 11)
    cf = plt.contourf(xg, yg, p, levels=levels, alpha=0.8, cmap=cmap, zorder=1, vmin=0, vmax=1)
    plt.colorbar(cf, ticks=levels, fraction=0.03, pad=0.04)


def _scrape(url: str, proxies: Dict[str, str]) -> str:
    return requests.get(url, proxies=proxies).text


def _one_hot_encode(data: np.ndarray, columns: List[int]) -> np.ndarray:
    columns = sorted(set(columns))[::-1]

    def ensure_matrix(x: np.ndarray) -> bool:
        return x if x.ndim == 2 else np.array(x).reshape(-1, 1)

    for c in columns:
        one_hot = LabelBinarizer().fit_transform(data[:, c])
        data = np.hstack((
            ensure_matrix(data[:, :c]),
            ensure_matrix(one_hot),
            ensure_matrix(data[:, c+1:])
        ))

    return data
