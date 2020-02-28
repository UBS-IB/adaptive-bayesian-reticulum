from typing import Union

import numpy as np
from scipy.special import betaln, gammaln, digamma

ARRAY_OR_FLOAT = Union[np.ndarray, float]


def multivariate_betaln(alphas: np.ndarray) -> ARRAY_OR_FLOAT:
    if len(alphas) == 2:
        return betaln(alphas[0], alphas[1])
    else:
        # see https://en.wikipedia.org/wiki/Beta_function#Multivariate_beta_function
        return np.sum([gammaln(alpha) for alpha in alphas], axis=0) - gammaln(alphas.sum())


def compute_log_p_data(
        prior: np.ndarray,
        k: Union[np.ndarray, float],
        betaln_prior: float) -> Union[np.ndarray, float]:
    # see https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall06/reading/bernoulli.pdf, equation (42)
    # which can be expressed as a fraction of beta functions
    posterior = prior + k
    return multivariate_betaln(posterior.T) - betaln_prior


def d_log_multivariate_beta_d_alphas(alphas: np.ndarray) -> ARRAY_OR_FLOAT:
    return digamma(alphas) - digamma(np.sum(alphas))


def sigmoid(x: ARRAY_OR_FLOAT) -> ARRAY_OR_FLOAT:
    # avoid underflow/overflow
    LOW = -708
    HIGH = 40

    if np.isscalar(x):
        return 1 if x > HIGH else 0 if x < LOW else 1 / (1 + np.exp(-x))
    else:
        lt_40 = x <= HIGH
        gt_m708 = x >= LOW
        if np.all(lt_40) and np.all(gt_m708):
            return 1 / (1 + np.exp(-x))
        else:
            result = np.zeros(x.shape)
            result[~lt_40] = 1
            valid = np.where(lt_40 & gt_m708)[0]
            result[valid] = 1 / (1 + np.exp(-x[valid]))
            return result


def pick_proportional(relative_probabilities: np.ndarray):
    # from page 161 of
    # https://github.com/Grant6899/books/blob/master/%5BMark%20Joshi%5DQuant%20Job%20Interview%20Questions%20And%20Answers.pdf
    p = relative_probabilities / np.sum(relative_probabilities)
    u = np.random.uniform(0, 1)
    i = 0
    while u > p[i]:
        u -= p[i]
        i += 1

    return i
