def chunkn(seq, n):
    """Makes n chunks from a seq, each about the same size.
    """
    size = len(seq) / n
    result = []
    last = 0.0

    while last < len(seq):
        result.append(seq[int(last):int(last + size)])
        last += size

    return result
