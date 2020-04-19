import numpy as np

def dtw(x, y, dist, warp = 1, w = 'inf'):
    '''Dynamic Time Warping of two sequences, accelerated with
    matrix operations instead of sequential vector operations

    Parameters
    ----------
    x: numpy.ndarray
        x.ndim == 1 or 2
    y: numpy.ndarray
        y.ndim == 1 or 2 (consistent with x)
    dist: function
        function which measures distance between frames from x and y
        if (ndim == 1)
            function should compare entry to entry(x[i], y[j]), and return single entry
        if (ndim == 2)
            function should compare vector to whole matrix(x[i], y), and return single vector
            it compares single vector frame from x to all vector frames in y.
            ex) x.shape == (T1, 10)
                y.shape == (T2, 10)
                dist(x[i], y)    ===> How dist is used
                dist(x[i], y).shape == (T2,)

    warp: maximum warp amount when finding path
        (Not Supported Yet)
    w: window size when computing cost matrix
        (Not Supported Yet)

    Returns
    -------
    distance: minimum sum of distance across all frames
    path: DTW path of x, y
    '''
    assert x.ndim == y.ndim, 'The number of dimensions do not match\n x: %s, y: %s '%(x.ndim, y.ndim)
    assert x.ndim < 3, 'Maximum number of dimensions are 2\n, ndim: %s'%(x.dim)
    # 0] Preparation
    T1 = len(x)
    T2 = len(y)
    swap = False
    if T1 > T2:
        print('swapped!')
        swap = True
        x, y = y, x
        T1, T2 = T2, T1
    # always T1 < T2, to accelerate calculation
    cost_matrix_ = np.zeros((T1+1, T2+1))
    cost_matrix_[1:, 0] = float('inf')
    cost_matrix_[0, 1:] = float('inf')
    cost_matrix = cost_matrix_[1:, 1:] # View

    # 1] Get cost matrix: dist(vector, matrix) -> Faster calculation
    # Scalar sequences
    if x.ndim == 1:
        cost_matrix[:, :] = dist(np.expand_dims(x,-1), np.broadcast_to(y, (T1,T2)))
        # cost_matrix = (x - y.expand(T1, -1).T).T
    # Vector sequences
    else:
        for i, vector in enumerate(x):
            cost_matrix[i] = dist(vector, y)
    # cost_matrix_base = cost_matrix.copy() # To save cost matrix

    # 2] Compute accumulated cost path(minimum cost matrix) & minimum cost path (T2 ~ T1+T2-1)
    # Accumulate to existing cost_matrix
    path_matrix = np.zeros((T1, T2, 2), dtype=int)
    x_i = {0:-1, 1:0, 2:-1}
    y_i = {0:-1, 1:-1, 2:0}
    for i in range(T1):
        for j in range(T2):
            search_list = [ cost_matrix_[i,j], cost_matrix_[i+1,j], cost_matrix_[i,j+1] ]
            min_hash = np.argmin(search_list)
            path_matrix[i,j] = i + x_i[min_hash] , j + y_i[min_hash]
            cost_matrix[i,j] += search_list[min_hash]

    # 3] Back-Trace minimum cost path
    i, j = T1-1, T2-1
    x_path = list()
    y_path = list()
    while (j>0 or i>0): # Since T1 < T2, i=0 occurs more. So check j first
        i, j = path_matrix[i,j]
        x_path.append(i)
        y_path.append(j)
    x_path.reverse()
    y_path.reverse()
    if swap == True:
        x_path, y_path = y_path, x_path
    return cost_matrix[-1,-1], [x_path, y_path]
