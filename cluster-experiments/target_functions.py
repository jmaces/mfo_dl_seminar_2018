#!/usr/bin/env python3
import numpy as np
from scipy.special import binom

def bernstein(x, k, n):
    ''' 1D Bernstein polynomial

    Computes the value of the k-th Bernstein polynomial of degree n at x.

    Args:
        x (:obj:`numpy array`): positions at which to evaluate the
            Bernstein polynomial
        k (int): The number of the Bernstein polynomial. Has to satisfy k >= 0.
        n (int): The degree of the Bernstein polynomial. Has to satisfy n >= k.

    Returns:
        :obj:`numpy array` of the same shape as x.
            The value of the k-th Berstein polynomial of degree n evaluated at
            locations x.

    Raises:
        ValueError: If 0 <= k <= n is not satisfied.

    '''
    if k < 0 or n < k:
        raise ValueError(
            'Bernstein polynomials only defined for 0 <= k <= n. '
            'Got (k,n)=({}, {}) instead'.format(k, n)
        )
    return binom(n, k)*(x**k)*((1-x)**(n-k))


def bernstein2d(x, y, k, n):
    ''' 2D Bernstein polynomial

    Computes the value of the tensor product of two Bernstein polynomials at
    positions (x, y).
    The k1-th Bernstein polynomial of degree n1 is used in the x dimension.
    The k2-th Bernstein polynomial of degree n2 is used in the y dimension.

    Args:
        x (:obj:`numpy array`): x positions at which to evaluate the
            Bernstein polynomial. Must be same shape as y.

            Usually x will be one of the return values of a call to meshgrid.
        y (:obj:`numpy array`): y positions at which to evaluate the
            Bernstein polynomial. Must be the same shape as x.

            Usually y will be one of the return values of a call to meshgrid.
        k (list or tuple of int): The numbers of the Bernstein polynomials.
            Given as a list or tuple (k1, k2) for the x and y dimension.
            Has to satisfy k >= 0 componentwise.
        n (list or tuple of int): The degrees of the Bernstein polynomials.
            Given as a list or tuple (n1, n2) for the x and y dimension.
            Has to satisfy n >= k componentwise.

    Returns:
        :obj:`numpy array` of the same shape as x and y.
            The value of the tensor product of the two Berstein polynomials
            evaluated at locations (x, y).

    Raises:
        ValueError: If 0 <= k <= n is not satisfied componentwise.

    '''
    if np.any(np.asarray(k) < 0) or np.any(np.asarray(n) < k):
        raise ValueError(
            'Bernstein polynomials only defined for 0 <= k <= n. '
            'Got (k1, k2)=({}, {}) and (n1, n2)=({}, {}) instead'
            .format(k[0], k[1], n[0], n[1])
        )
    return bernstein(x, k[0], n[0])*bernstein(y, k[1], n[1])
    
def signpoly(x, n):
    ''' 1D signed polynomial

    Computes the value of the n-th anti-derivative of sgn(x) at x.
    The anti-derivative is given by the expression 1/n! x^n sign(x).
    By construction this function is (n-1)-times continuously differentiable
    and a polynomial of degree n.

    Args:
        x (:obj:`numpy array`): x positions at which to evaluate the
            signed polynomial.
        n (int): The degree/regularity of the signed polynomial. 
            Has to satisfy n >= 0.
            (The function is of degree n and in the space C^(n-1).)

    Returns:
        :obj:`numpy array` of the same shape as x.
            The value of the signed polynomial of degree n
            evaluated at locations x.

    Raises:
        ValueError: If 0 <= n is not satisfied.


    '''
    if n < 0:
        raise ValueError(
            'The signed polynomial is only defined for 0 <= n. '
            'Got n={} instead'.format(n)
        )
    return 1/np.math.factorial(n)*np.power(x, n)*np.sign(x)
    
def signpoly2d(x, y, n):
    ''' 2D Bernstein polynomial

    Computes the value of the tensor product of two signed polynomials at
    positions (x, y).
    
    The signed polynomial of degree n1 is used in the x dimension.
    The signed polynomial of degree n2 is used in the y dimension.
    
    The signed polynomials are given by the expression 1/n! x^n sign(x).
    By construction this function is (n-1)-times continuously differentiable
    and a polynomial of degree n.

    Args:
        x (:obj:`numpy array`): x positions at which to evaluate the
            signed polynomial. Must be same shape as y.

            Usually x will be one of the return values of a call to meshgrid.
        y (:obj:`numpy array`): y positions at which to evaluate the
            signed polynomial. Must be the same shape as x.

            Usually y will be one of the return values of a call to meshgrid.
        n (list or tuple of int): The degrees of the signed polynomials.
            Given as a list or tuple (n1, n2) for the x and y dimension.
            Has to satisfy n >= 0 componentwise.

    Returns:
        :obj:`numpy array` of the same shape as x and y.
            The value of the tensor product of the two signed polynomials
            evaluated at locations (x, y).

    Raises:
        ValueError: If 0 <= n is not satisfied componentwise.

    '''
    if np.any(np.asarray(n) < 0):
        raise ValueError(
            'Signed polynomials only defined for 0 <= n. '
            'Got (n1, n2)=({}, {}) instead'
            .format(n[0], n[1])
        )
    return signpoly(x, n[0])*signpoly(y, n[1])
