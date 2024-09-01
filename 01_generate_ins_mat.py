import os
import argparse
import numpy as np
import scipy.sparse
import time

def generate_setcover(nrows, ncols, density, filename, rng, max_coef=100):
    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """
    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = rng.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = rng.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = rng.choice(nrows, size=n, replace=False)
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = rng.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = rng.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()

    # save matrix
    Amat = np.concatenate((c[np.newaxis, :], A.todense()), axis=0)
    txtname = filename[:-3]
    np.savetxt(f'{txtname}.txt', Amat, fmt = "%d", delimiter=' ')

    indices = A.indices
    indptr = A.indptr

    # write problem
    with open(filename, 'w') as file:
        file.write("minimize\nOBJ:")
        file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))

        file.write("\n\nsubject to\n")
        for i in range(nrows):
            row_cols_str = "".join([f" +1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
            file.write(f"C{i}:" + row_cols_str + f" >= 1\n")

        file.write("\nbinary\n")
        file.write("".join([f" x{j+1}" for j in range(ncols)]))



if __name__ == '__main__':
    # small case
    nrows = 20
    ncols = 20
    dens = 0.1
    max_coef = 100

    filenames = []
    nrowss = []
    ncolss = []
    denss = []

    n = 100
    lp_dir = f'/home/dingzhenxin/workspace/DPexpc/data/setcover_{nrows}r_{ncols}c_{dens}d'
    print(f'{n} instances in {lp_dir}')
    os.makedirs(lp_dir, exist_ok=True)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nrowss.extend([nrows] * n)
    ncolss.extend([ncols] * n)
    denss.extend([dens] * n)

    rng = np.random.RandomState(0)

    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)


    # smiddle case
    nrows = 50
    ncols = 50
    dens = 0.1
    max_coef = 500

    filenames = []
    nrowss = []
    ncolss = []
    denss = []

    n = 100
    lp_dir = f'/home/dingzhenxin/workspace/DPexpc/data/setcover_{nrows}r_{ncols}c_{dens}d'
    print(f'{n} instances in {lp_dir}')
    os.makedirs(lp_dir, exist_ok=True)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nrowss.extend([nrows] * n)
    ncolss.extend([ncols] * n)
    denss.extend([dens] * n)

    rng = np.random.RandomState(0)

    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)

  # middle case
    nrows = 100
    ncols = 100
    dens = 0.1
    max_coef = 500

    filenames = []
    nrowss = []
    ncolss = []
    denss = []

    n = 100
    lp_dir = f'/home/dingzhenxin/workspace/DPexpc/data/setcover_{nrows}r_{ncols}c_{dens}d'
    print(f'{n} instances in {lp_dir}')
    os.makedirs(lp_dir, exist_ok=True)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nrowss.extend([nrows] * n)
    ncolss.extend([ncols] * n)
    denss.extend([dens] * n)

    rng = np.random.RandomState(0)

    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)


    # big case
    nrows = 500
    ncols = 1000
    dens = 0.05
    max_coef = 100

    filenames = []
    nrowss = []
    ncolss = []
    denss = []

    n = 10
    lp_dir = f'/home/dingzhenxin/workspace/DPexpc/data/setcover_{nrows}r_{ncols}c_{dens}d'
    print(f'{n} instances in {lp_dir}')
    os.makedirs(lp_dir, exist_ok=True)
    filenames.extend([os.path.join(lp_dir, f'instance_{i+1}.lp') for i in range(n)])
    nrowss.extend([nrows] * n)
    ncolss.extend([ncols] * n)
    denss.extend([dens] * n)

    rng = np.random.RandomState(0)

    for filename, nrows, ncols, dens in zip(filenames, nrowss, ncolss, denss):
        print(f'  generating file {filename} ...')
        generate_setcover(nrows=nrows, ncols=ncols, density=dens, filename=filename, rng=rng, max_coef=max_coef)






































