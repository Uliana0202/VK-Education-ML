import numpy as np


def product_of_diagonal_elements_vectorized(matrix: np.array):
    diagonal = np.diag(matrix)
    return np.prod(diagonal[diagonal != 0])


def are_equal_multisets_vectorized(x: np.array, y: np.array):
    return np.array_equal(np.sort(x), np.sort(y))


def max_before_zero_vectorized(x: np.array):
    after_zeros = np.where(x == 0)[0] + 1
    if after_zeros[-1] == x.shape[0]:
        after_zeros = after_zeros[:-1]
    return np.max(x[after_zeros])


def add_weighted_channels_vectorized(image: np.array):
    weights = np.array([0.299, 0.587, 0.114])
    return np.dot(image[..., :3], weights)


def run_length_encoding_vectorized(x: np.array):
    changes = np.where(np.diff(x) != 0)[0]
    indices = np.concatenate(([0], changes + 1, [x.shape[0]]))
    return x[indices[:-1]], np.diff(indices)
