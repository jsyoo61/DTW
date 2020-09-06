# DTW
Fast dtw implementation with matrix operations
Dynamic Time Warping of two sequences, accelerated with
    matrix operations instead of sequential vector operations
Faster than: dtw, fastdtw package (loops through iterations without matrix operations)

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
    path: DTW path of x, y in list form
