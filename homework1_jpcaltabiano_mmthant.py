import numpy as np

def problem1 (A, B):
    return A + B

def problem2 (A, B, C):
    return A.dot(B) - C

def problem3 (A, B, C):
    return (A * B) + C.T

def problem4 (x, y):
    return x.T.dot(y)

def problem5 (A):
    return (np.zeros(A.shape))

def problem6 (A):
    return (np.ones(A.shape))

# TODO: Unsure if this is correct
def problem7 (A, alpha):
    i = np.eye(A.shape[0], A.shape[1])
    return A + (alpha * i)

def problem8 (A, i, j):
    return A[i,j]

def problem9 (A, i):
    return A[i].sum()

def problem10 (A, c, d):
    # the following is identical to i.e. 'A[np.nonzero(c <= A)]'
    A = A[c <= A]
    A = A[d >= A]
    return A.mean()

# TODO: Unsure if this is correct
def problem11 (A, k):
    evals, evecs = np.linalg.eig(A)
    idx = [np.argsort(evals)[-k:]]
    return evecs[:, idx].reshape(A.shape[0], k)

def problem12 (A, x):
    return np.linalg.solve(A, x)

def problem13 (A, x):
    return np.linalg.solve(x, A)
