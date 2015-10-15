import fileinput
import os


def chunkn(seq, n):
    """Makes n chunks from a seq, each about the same size.
    """
    size = len(list(seq)) / n
    result = []
    last = 0.0

    i = 0
    while last < len(seq):
        yield i, seq[int(last):int(last + size)]
        last += size
        i += 1


def add_header(header, filename):
    """Adds a header line to an existing file.
    """
    for line in fileinput.input([filename], inplace=True):
        if fileinput.isfirstline():
            print(header)
        print(line, end='')
