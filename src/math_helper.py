import numpy as np


def partial_derivative_x(vector_field, element_length):
    diff = np.zeros_like(vector_field)

    diff[1:-1, 1:-1] = (
        (vector_field[2:  , 1:-1] - vector_field[0:-2, 1:-1]) / (2 * element_length)
    )

    return diff

def partial_derivative_y(vector_field, element_length):
    diff = np.zeros_like(vector_field)

    diff[1:-1, 1:-1] = (
        (vector_field[1:-1, 2:  ] - vector_field[1:-1, 0:-2]) / (2 * element_length)
    )

    return diff

def gradient(field, element_length):
    gradient_applied = np.concatenate(
        (
            partial_derivative_x(field, element_length)[..., np.newaxis],
            partial_derivative_y(field, element_length)[..., np.newaxis],
        ),
        axis=-1,
    )

    return gradient_applied

def divergence(vector_field, element_length):
    divergence_applied = (
        partial_derivative_x(vector_field[..., 0], element_length)
        +
        partial_derivative_y(vector_field[..., 1], element_length)
    )

    return divergence_applied

def laplace(field, element_length):
    diff = np.zeros_like(field)

    diff[1:-1, 1:-1] = (
        (
            field[0:-2, 1:-1]
            +
            field[1:-1, 0:-2]
            - 4 *
            field[1:-1, 1:-1]
            +
            field[2:  , 1:-1]
            +
            field[1:-1, 2:  ]
        ) / (
            element_length**2
        )
    )

    return diff
