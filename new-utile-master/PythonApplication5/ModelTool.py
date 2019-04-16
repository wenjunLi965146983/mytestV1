'''
Some tool
for DataGenerator

Edit by LWJ 2019-4-10

'''

from functools import reduce


def compose(*funcs):
    """
    Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x:reduce(lambda v,f:f(v),funcs,x)
    # return a new function:f(V(x))
    if funcs: 
        return reduce(lambda f,g:lambda *a,**kw:g(f(*a,**kw)),funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')



