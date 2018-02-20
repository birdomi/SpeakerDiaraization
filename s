def sparse_tuple_from(sequences, dtype=np.int32):
    r"""Creates a sparse representention of ``sequences``.
    Args:
        * sequences: a list of lists of type dtype where each element is a sequence

    Returns a tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int32)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1]+1], dtype=np.int32)

    return indices,values,shape
