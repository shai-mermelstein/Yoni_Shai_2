import mykmeanssp

import sys
import numpy as np
import pandas as pd

DEFAULT_MAX_ITER = 300
INVALID_INPUT_MSG = 'Invalid Input!'
ERROR_MSG = 'An Error Has Occurred'

class InvalidInput(Exception):
    pass

def get_input():
    """returns k, max_iter, epsilon, file_name_1, file_name_2"""
    args = sys.argv[1:]
    if len(args) == 4:
        args.insert(1, DEFAULT_MAX_ITER)
    assert len(args) == 5
    assert args[3].endswith(('.txt', '.csv')) and args[4].endswith(('.txt', '.csv'))
    k, max_iter, epsilon = int(args[0]), int(args[1]), float(args[2])
    assert k > 0 and max_iter > 0 and epsilon >= 0
    return k, max_iter, epsilon, args[3], args[4]

def load_and_join_vectors(file_name_1: str, file_name_2: str):
    """
    returns inner join of given files using the first column in each file as key, 
    as well as the number of vectors, their dimension, and given indices.
    """
    df1 = pd.read_csv(file_name_1, header = None, index_col = 0)
    df2 = pd.read_csv(file_name_2, header = None, index_col = 0)
    df = pd.merge(df1, df2, how='inner', left_index=True, right_index=True, sort=True)

    arr = np.require(df.to_numpy(), requirements=['C', 'A'])
    numV, d = arr.shape
    indices = df.index.to_numpy(int)

    return arr, numV, d, indices

def kmeans_pp(vectors: np.ndarray, numV: int, d: int, k: int):
    """runs k-means++ algorithms and returns its result"""
    np.random.seed(0)
    res = [ np.random.choice(numV) ]
    norm_squared = lambda v: np.inner(v, v)

    for _ in range(1, k):
        P = np.fromiter((min(norm_squared(vectors[l] - vectors[j]) for j in res) \
            for l in range(numV)), float) 
        P = P / np.sum(P)
        res.append(np.random.choice(numV, p=P))
    
    return res

def main():
    try:
        try:
            k, max_iter, epsilon, file_name_1, file_name_2 = get_input()
        except:
            raise InvalidInput
        vectors, numV, d, indices = load_and_join_vectors(file_name_1, file_name_2)
        if k >= numV:
            raise InvalidInput
        initial_centroids = kmeans_pp(vectors, numV, d, k)
        res = vectors[initial_centroids]

        print(','.join(str(indices[j]) for j in initial_centroids))
        assert mykmeanssp.fit(k, max_iter, epsilon, vectors, numV, d, res) == 0
        for v in res: 
            print(','.join('{:.4f}'.format(x) for x in v))

    except InvalidInput:
        sys.exit(INVALID_INPUT_MSG)
    except:
        sys.exit(ERROR_MSG)


if __name__ == '__main__':
    main()
    