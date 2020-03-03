# Loads the Ripley dataset and trains the Adaptive Bayesian Reticulum model on it.
# Produces plots and some performance statistics.
#
# If you are behind a corporate proxy you need to specify the proxies by supplying
# the following environment variables:
# - HTTP_PROXY in the form:  http://user:password@your-http-proxy:port
# - HTTPS_PROXY in the form: https://user:password@your-https-proxy:port

import datetime as dt
import os

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

from examples.util import load_ripley, plot_2d_hyperplane
from reticulum import AdaptiveBayesianReticulum

# read proxies from environment variables (if required, see comment at the top of this file)
proxies = {
    'http': os.environ.get('HTTP_PROXY', None),
    'https': os.environ.get('HTTPS_PROXY', None)
}

# load data
train, test = load_ripley(proxies)

# extract input and target
X_train = train[:, :-1]
y_train = train[:, -1]
X_test = test[:, :-1]
y_test = test[:, -1]

# train model
model = AdaptiveBayesianReticulum(
    prior=(1, 1),
    pruning_factor=1.05,
    n_iter=40,
    learning_rate_init=0.1,
    n_gradient_descent_steps=100,
    initial_relative_stiffness=2,
    random_state=42)

t0 = dt.datetime.utcnow()
model.fit(X_train, y_train, verbose=True)
t1 = dt.datetime.utcnow()
print('Model:')
print(model)
print(f'Training took {t1-t0}')

# evaluate performance
log_loss_train = log_loss(y_train, model.predict_proba(X_train))
log_loss_test = log_loss(y_test, model.predict_proba(X_test))
accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))
info_train = f'Train: Log-loss = {log_loss_train}, accuracy = {100*accuracy_train:.4f} %'
info_test = f'Test: Log-loss = {log_loss_test}, accuracy = {100*accuracy_test:.4f} %'
print(f'Depth:  {model.get_depth()}')
print(f'Leaves: {model.get_n_leaves()}')
print(info_train)
print(info_test)
print(f'Feature importance: {model.feature_importance()}')

# plot
plot_2d_hyperplane(model, X_train, y_train, info_train, X_test, y_test, info_test)
