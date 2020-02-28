# Generates the Ripley evolution charts of the paper.
#
# If you are behind a corporate proxy you need to specify the proxies by supplying
# the following environment variables:
# - HTTP_PROXY in the form:  http://user:password@your-http-proxy:port
# - HTTPS_PROXY in the form: https://user:password@your-https-proxy:port

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional, Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np

from reticulum import AdaptiveBayesianReticulum
from reticulum.node import Node
from examples.util import load_ripley

seed = 42
n_jobs = 5
font_size = 14
title_size = 20
figure_suffixes = ['.pdf', '.eps']
step_index = [0]
show_plot = True
figure_in_progress = [False]
cmap = plt.get_cmap('tab20')
show_pruning_step = True


# a line segment for plotting
class Line:
    def __init__(self, p0: Union[List, np.ndarray], p1: Union[List, np.ndarray]) -> None:
        if p0[0] > p1[0]:
            p1, p0 = p0, p1

        self.p0 = np.asarray(p0)
        self.p1 = np.asarray(p1)

    def intersect(self, other: Line) -> Optional[np.ndarray]:
        da = self.p1-self.p0
        ma = da[1]/da[0]

        db = other.p1-other.p0
        mb = db[1]/db[0]

        x0a = self.p0[0]
        x1a = self.p1[0]
        x0b = other.p0[0]
        x1b = other.p1[0]
        y0a = self.p0[1]
        y0b = other.p0[1]

        x = (y0a-y0b + mb*x0b-ma*x0a) / (mb-ma)
        y = y0a + ma*(x-x0a)

        if x0a <= x <= x1a and x0b <= x <= x1b:
            return np.array([x, y])
        else:
            return None

    def plot(self, *args: Any, **kwargs: Any) -> None:
        plt.plot([self.p0[0], self.p1[0]], [self.p0[1], self.p1[1]], *args, **kwargs)

    def __str__(self) -> str:
        return f'{self.p0} -> {self.p1}'


@dataclass
class Parent:
    line: Line
    weights: np.ndarray
    side: str


# plots the root node split and all child nodes recursively
def plot_root(root: Node, title: str) -> None:
    plt.title(title)

    plt.plot(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 'b.', ms=3)
    plt.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 'r.', ms=3)

    def plot_node(node: Node, node_vs_color: Dict={}, level: int=0, parents: List[Parent]=[], side: str=None) -> None:
        # pick an arbitrary origin and get the normal
        w = node.weights
        origin = w[1:] * -w[0] / np.dot(w[1:], w[1:])
        normal = w[1:]

        # construct line segment
        m = -normal[0]/normal[1]
        y0 = origin[1] + m*(x_min-origin[0])
        y1 = origin[1] + m*(x_max-origin[0])

        # raw line without intersections
        line = Line([x_min, y0], [x_max, y1])

        # intersect with parents
        for parent in parents:
            p = line.intersect(parent.line)
            if p is not None:
                # determine side of line to keep
                activation0 = parent.weights[0] + np.dot(line.p0, parent.weights[1:])

                if (parent.side == 'L' and activation0 > 0) or (parent.side == 'R' and activation0 < 0):
                    line = Line(line.p0, p)
                else:
                    line = Line(p, line.p1)

        # intersect with top/bottom
        p = line.intersect(top)
        if p is not None:
            if y0 > y_max:
                line = Line(p, line.p1)
            else:
                line = Line(line.p0, p)

        p = line.intersect(bottom)
        if p is not None:
            if y0 < y_min:
                line = Line(p, line.p1)
            else:
                line = Line(line.p0, p)

        # generate line name
        if side is not None:
            side_name = ' - '.join(f'{parents[i].side}{level-len(parents)+i+1}' for i in range(len(parents)))
        else:
            side_name = ''

        side_name = 'Root' if len(side_name) == 0 else 'Root - ' + side_name

        # make sure node colors don't change
        if id(node) not in node_vs_color:
            color = cmap(len(node_vs_color))
            node_vs_color[id(node)] = color
        else:
            color = node_vs_color[id(node)]

        # compute line width as a function of the stiffness
        stiffness = np.linalg.norm(normal)
        lw = 1 + 50/stiffness

        line.plot(color=color, label=side_name, lw=lw, alpha=0.7)

        if node.left__child:
            plot_node(node.left__child, node_vs_color, level+1, parents=parents + [Parent(line, w, 'L')], side='L')

        if node.right_child:
            plot_node(node.right_child, node_vs_color, level+1, parents=parents + [Parent(line, w, 'R')], side='R')

    plot_node(root)


# read proxies from environment variables (if required, see comment at the top of this file)
proxies = {
    'http': os.environ.get('HTTP_PROXY', None),
    'https': os.environ.get('HTTPS_PROXY', None)
}

# load Ripley data
train = load_ripley(proxies)[0]
X_train = train[:, :-1]
y_train = train[:, -1]

# find data boundaries
x_min = X_train[:, 0].min()
x_max = X_train[:, 0].max()
y_min = X_train[:, 1].min()
y_max = X_train[:, 1].max()

top = Line([x_min, y_max], [x_max, y_max])
bottom = Line([x_min, y_min], [x_max, y_min])


def create_figure() -> None:
    plt.figure(figsize=[10, 2.5])
    figure_in_progress[0] = True


def subplot(n: int) -> None:
    plt.subplot(n)
    plt.gca().set_aspect(1)
    plt.gca().axis('off')
    yr = y_max-y_min
    plt.ylim(y_min-0.15*yr, y_max+0.15*yr)


def try_finalize_figure() -> None:
    if not figure_in_progress[0]:
        return

    # plt.tight_layout()

    for figure_suffix in figure_suffixes:
        plt.gcf().savefig(f'ripley_evolution_{step_index[0]}' + figure_suffix)

    if show_plot:
        plt.show()

    figure_in_progress[0] = False


# callback to record model state during training
def callback(event: str, root: Node, node: Node) -> None:
    is_root = node is root
    is_local = (is_root and root.left__child is None and root.right_child is None) or not is_root

    if event == 'gd_start':
        if is_local:
            try_finalize_figure()
            step_index[0] += 1

            if step_index[0] <= 3 or step_index[0] == 18:
                create_figure()
                subplot(141 if show_pruning_step else 131)
                plot_root(root, 'Created Root Node' if is_root else 'Created Node')
    elif event == 'gd_end':
        if is_local:
            subplot(142 if show_pruning_step else 132)
            plot_root(root, 'Local GD')
        else:
            subplot(143 if show_pruning_step else 133)
            plot_root(root, 'Global GD')
    elif event == 'prune' and show_pruning_step:
            subplot(144)
            plot_root(root, 'After Pruning')


# model
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

# attach callback
model.callback = lambda event, root, node: callback(event, root, node)

# fit model (which will call callback)
model.fit(X_train, y_train)
plt.clf()

# plot final state
create_figure()
subplot(131)
plot_root(model.root_, 'Final model')
try_finalize_figure()

print(model)
