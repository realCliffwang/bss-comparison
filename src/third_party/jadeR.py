"""
JADE (Joint Approximate Diagonalization of Eigenmatrices) for real signals.

Source: https://github.com/gbeckers/jadeR (MIT License)
Original MATLAB: Jean-Francois Cardoso <cardoso@sig.enst.fr>
Python translation: Gabriel Beckers <gabriel@gbeckers.nl>

This is a zero-dependency single-file implementation. Copy to src/third_party/
to avoid external package dependencies.

Reference:
  Cardoso, J. (1999) "High-order contrasts for independent component
  analysis." Neural Computation, 11(1): 157-192.
"""
from __future__ import print_function

import sys
import os
import getopt
from numpy import abs, append, arange, arctan2, argsort, array, concatenate, \
    cos, diag, dot, eye, float64, loadtxt, matrix, multiply, ndarray, \
    savetxt, sign, sin, sqrt, zeros
from numpy.linalg import eig, pinv

__version__ = 1.9


def jadeR(X, m=None, verbose=False):
    """
    Blind separation of real signals with JADE.

    JADE implements Independent Component Analysis (ICA) developed by
    Jean-Francois Cardoso. See:
      Cardoso, J. (1999) High-order contrasts for independent component
      analysis. Neural Computation, 11(1): 157-192.

    Translated into NumPy from original Matlab Version 1.8 (May 2005) by
    Gabriel Beckers, http://gbeckers.nl .
    Corrections by David Rivest-Henault to match jadeR.m at machine precision.

    Parameters
    ----------
    X : ndarray (n, T)
        Data matrix: n sensors, T samples.
    m : int or None
        Number of independent components to extract.
        Defaults to None (m == n).
    verbose : bool
        Print progress info. Default is False.

    Returns
    -------
    B : ndarray (m, n)
        Demixing matrix. Separated sources: Y = B @ X.
        Rows of B are ordered so columns of pinv(B) have decreasing norm,
        placing the most energetically significant components first.
    """
    assert isinstance(X, ndarray), \
           "X (input data matrix) is of the wrong type (%s)" % type(X)
    origtype = X.dtype
    X = matrix(X.astype(float64))
    assert X.ndim == 2, "X has %d dimensions, should be 2" % X.ndim
    assert (verbose == True) or (verbose == False), \
           "verbose parameter should be either True or False"

    [n, T] = X.shape
    assert n < T, "number of sensors must be smaller than number of samples"

    if m is None:
        m = n
    assert m <= n, \
        "number of sources (%d) is larger than number of sensors (%d)" % (m, n)

    if verbose:
        print("jade -> Looking for %d sources" % m)
        print("jade -> Removing the mean value")
    X -= X.mean(1)

    if verbose:
        print("jade -> Whitening the data")
    [D, U] = eig((X * X.T) / float(T))
    k = D.argsort()
    Ds = D[k]
    PCs = arange(n - 1, n - m - 1, -1)

    B = U[:, k[PCs]].T
    scales = sqrt(Ds[PCs])
    B = diag(1. / scales) * B
    X = B * X

    del U, D, Ds, k, PCs, scales

    if verbose:
        print("jade -> Estimating cumulant matrices")

    X = X.T
    dimsymm = int((m * (m + 1)) / 2)
    nbcm = int(dimsymm)
    CM = matrix(zeros([m, m * nbcm], dtype=float64))
    R = matrix(eye(m, dtype=float64))
    Qij = matrix(zeros([m, m], dtype=float64))
    Xim = zeros(m, dtype=float64)
    Xijm = zeros(m, dtype=float64)

    Range = arange(m)

    for im in range(m):
        Xim = X[:, im]
        Xijm = multiply(Xim, Xim)
        Qij = multiply(Xijm, X).T * X / float(T) - R - 2 * (R[:, im] * R[:, im].T)
        CM[:, Range] = Qij
        Range = Range + m
        for jm in range(im):
            Xijm = multiply(Xim, X[:, jm])
            Qij = sqrt(2) * (multiply(Xijm, X).T * X / float(T)
                             - R[:, im] * R[:, jm].T - R[:, jm] * R[:, im].T)
            CM[:, Range] = Qij
            Range = Range + m

    V = matrix(eye(m, dtype=float64))

    Diag = zeros(m, dtype=float64)
    On = 0.0
    Range = arange(m)
    for im in range(nbcm):
        Diag = diag(CM[:, Range])
        On = On + (Diag * Diag).sum(axis=0)
        Range = Range + m
    Off = (multiply(CM, CM).sum(axis=0)).sum(axis=1) - On
    seuil = 1.0e-6 / sqrt(T)
    encore = True
    sweep = 0
    updates = 0
    upds = 0
    g = zeros([2, nbcm], dtype=float64)
    gg = zeros([2, 2], dtype=float64)
    G = zeros([2, 2], dtype=float64)
    c = 0
    s = 0
    ton = 0
    toff = 0
    theta = 0
    Gain = 0

    if verbose:
        print("jade -> Contrast optimization by joint diagonalization")

    while encore:
        encore = False
        if verbose:
            print("jade -> Sweep #%3d" % sweep)
        sweep = sweep + 1
        upds = 0
        Vkeep = V

        for p in range(m - 1):
            for q in range(p + 1, m):
                Ip = arange(p, m * nbcm, m)
                Iq = arange(q, m * nbcm, m)

                g = concatenate([CM[p, Ip] - CM[q, Iq], CM[p, Iq] + CM[q, Ip]])
                gg = dot(g, g.T)
                ton = gg[0, 0] - gg[1, 1]
                toff = gg[0, 1] + gg[1, 0]
                theta = 0.5 * arctan2(toff, ton + sqrt(ton * ton + toff * toff))
                Gain = (sqrt(ton * ton + toff * toff) - ton) / 4.0

                if abs(theta) > seuil:
                    encore = True
                    upds = upds + 1
                    c = cos(theta)
                    s = sin(theta)
                    G = matrix([[c, -s], [s, c]])
                    pair = array([p, q])
                    V[:, pair] = V[:, pair] * G
                    CM[pair, :] = G.T * CM[pair, :]
                    CM[:, concatenate([Ip, Iq])] = \
                        append(c * CM[:, Ip] + s * CM[:, Iq],
                               -s * CM[:, Ip] + c * CM[:, Iq],
                               axis=1)
                    On = On + Gain
                    Off = Off - Gain

        if verbose:
            print("completed in %d rotations" % upds)
        updates = updates + upds

    if verbose:
        print("jade -> Total of %d Givens rotations" % updates)

    B = V.T * B

    if verbose:
        print("jade -> Sorting the components")

    A = pinv(B)
    keys = array(argsort(multiply(A, A).sum(axis=0)[0]))[0]
    B = B[keys, :]
    B = B[::-1, :]

    if verbose:
        print("jade -> Fixing the signs")
    b = B[:, 0]
    signs = array(sign(sign(b) + 0.1).T)[0]
    B = diag(signs) * B

    return B.astype(origtype)


class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg


def main(argv=None):
    """
    jadeR -- Blind separation of real signals with JADE for Python.

    Usage summary: python jadeR.py [options] inputfile

    Options:
      -h    Help.
      -m    Number of sources requested (defaults to number of sensors).
      -o    Output file name.
      -s    Silent mode.
      -t    Transpose data before processing.
    """
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "f:hm:o:st",
                    ["format=", "help", "m=", "outputfile=", "silent", "transpose"])
        except getopt.GetoptError as err:
            raise Usage(err)
    except Usage as err:
        sys.stderr.write(err.msg)
        sys.stderr.write("for help use --help\n")
        sys.exit(2)

    format = 'txt'
    m = None
    verbose = True
    outputfilename = None
    transpose = False

    try:
        for o, a in opts:
            if o in ("-h", "--help"):
                print(main.__doc__)
                sys.exit(0)
            elif o in ("-f", "--format"):
                if a not in ('txt'):
                    raise Usage("'%s' is not a valid input format\n" % a)
                else:
                    format = a
            elif o in ("-m", "--m"):
                try:
                    m = int(a)
                except:
                    raise Usage("m should be an integer\n")
            elif o in ("-o", "--outputfile"):
                outputfilename = a
            elif o in ("-s", "--silent"):
                verbose = False
            elif o in ("-t", "--transpose"):
                transpose = True
        if len(args) != 1:
            raise Usage("please provide one and only one input file\n")
        if not os.path.isfile(args[0]):
            raise Usage("%s is not a valid file name\n" % args[0])
        filename = args[0]
        if outputfilename is None:
            outputfilename = filename.split('.')[0] + '_jade' + '.txt'
        if os.path.exists(outputfilename):
            raise Usage("file %s already exists, bailing out\n" % outputfilename)
    except Usage as err:
        sys.stderr.write(err.msg)
        sys.stderr.write("for help use --help\n")
        sys.exit(2)
    if format == 'txt':
        if verbose:
            print("loading data from text file...")
        X = loadtxt(filename)
        if transpose == False:
            X = X.T
        if verbose:
            print("finished; found %d sensors, each having %d samples.\n"
                  % (X.shape[0], X.shape[1]))

        B = jadeR(X=X, m=m, verbose=verbose)
        Y = B * matrix(X)

        if verbose:
            print("\nsaving results to text file '%s' ..." % outputfilename)
        savetxt(outputfilename, Y.T)
        if verbose:
            print("finished!")


if __name__ == "__main__":
    sys.exit(main())
