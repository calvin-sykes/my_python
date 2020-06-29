import itertools as _it

def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...
    
    from https://docs.python.org/3/library/itertools.html
    """
    a, b = _it.tee(iterable)
    next(b, None)
    return zip(a, b)
