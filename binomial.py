from decimal import Decimal, getcontext
import numpy as np

def compute_binomial_coefficient_matrix(shape):
    """Using dynamic programming and the recursive property of binomial coefficients computes the matrix of binomial coefficients of a given shape of all combinations from range <0, shape>.

    Requires:
    shape (int) - defines a shape of the matrix and the range from the binomial coefficients will be computed

    Ensures:
    result_matrix (np.ndarray) - a matrix of the binomial coefficients"""
    getcontext().prec = 20
    result_matrix = np.zeros((shape, shape),dtype=Decimal)

    for n in range(shape):
        for k in range(n + 1): #filling only lower diagonal of the matrix, the upper is zero
            if k == 0 or k == n:
                result_matrix[n, k] = Decimal(1)
            else:
                result_matrix[n, k] = Decimal(result_matrix[n-1, k-1] + result_matrix[n-1, k])

    return result_matrix

def binomial_probs(probs):
    """Computes the probability distribution of observing the vector of copy numbers (x, y, z) in the daughter cell based on the a priori probability distribution of x, y and z in the mother
    cell. A Decimal module is required due to working with very large integers and very small floats.

    Requires:
    probs (np.ndarray) - a priori probability distribution of x, y and z in the mother cell

    Ensures:
    result_probs (np.ndarray) - a matrix of the probability distribution of observing the vector of copy numbers (x, y, z) in the daughter cell"""

    X_shape = probs.shape[0] 
    Y_shape = probs.shape[1]
    binom_coeffs_x = compute_binomial_coefficient_matrix(X_shape)
    binom_coeffs_y = compute_binomial_coefficient_matrix(Y_shape)
    powers_of_2_x = [Decimal(2) ** (-x) for x in range(X_shape)]
    powers_of_2_y = [Decimal(2) ** (-y) for y in range(Y_shape)]
    probs = np.vectorize(Decimal)(probs)
    daughter0 = binom_coeffs_x.T @ (np.diag(powers_of_2_x) @ probs[:,:,0] @ np.diag(powers_of_2_y)) @ binom_coeffs_y
    daughter1 = binom_coeffs_x.T @ (np.diag(powers_of_2_x) @ probs[:,:,1] @ np.diag(powers_of_2_y)) @ binom_coeffs_y
    new_probs = np.zeros(shape=probs.shape)
    new_probs[:,:,0] = daughter0 
    new_probs[:,:,1] = daughter1
    
    return new_probs