import numpy as np
import tensorflow as tf
import scipy.special as sp


def colon(a, b):
    return tf.reduce_sum(tf.multiply(a, b))


def nabla_hat(u_x):
    return 0.5*tf.math.add(u_x, tf.transpose(u_x))


def legendre_3d(c, v):
    """

    :param c: [int i, int j, int k] Vektor mit dem Grad dreier Legendre-Polynome
    :param v: [float x, float y, float z] 3D-Vektor, an dem die Polynome ausgewertet werden
    :return: P_i(x)*P_j(y)*P_k(z), Ableitungen des vorigen Polynoms
    """
    derivative_polys = [np.polyder(sp.legendre(i)) for i in c]
    x, y, z = sp.eval_legendre(c[0], 1-2*v[0]), sp.eval_legendre(c[1], 1-2*v[1]), sp.eval_legendre(c[2], 1-2*v[2])-1
    dx, dy, dz = (np.polyval(derivative_polys[0], v[0]), np.polyval(derivative_polys[1], v[1]),
                  np.polyval(derivative_polys[2], v[2]+1))
    value = x*y*z
    derivatives = [dx*y*z, x*dy*z, x*y*dz]
    return value, derivatives


def legendre_base(n, x):
    """
    Gibt 3D-zu-3D-Legendre-Polynome und deren Ableitungen an x zurück
    :param n: Grad, bis zu dem die Legendre-Polynome erzeugt werden.
    :param x: Punkt an dem ausgewertet wird
    :return: pols(x), [nabla * pols(x), nabla_hut(pols(x))]
    """

    indices = []
    for m in range(n+1):
        indices += find_3tuples(m)

    all_legendre = [legendre_3d(index, x) for index in indices]
    values = [legendre[0] for legendre in all_legendre]
    derivatives = [legendre[1] for legendre in all_legendre]

    # Gesonderte Polynome in z-Richtung wegen des Dirichlet-Randes

    functions = [np.array([legendre_1d, 0, 0], dtype=np.float32) for legendre_1d in values]
    functions += [np.array([0, legendre_1d, 0], dtype=np.float32) for legendre_1d in values]
    functions += [np.array([0, 0, legendre_1d], dtype=np.float32) for legendre_1d in values]

    z = [0, 0, 0]
    derivatives_3d = [np.array([legendre_1d, z, z], dtype=np.float32) for legendre_1d in derivatives]
    derivatives_3d += [np.array([z, legendre_1d, z], dtype=np.float32) for legendre_1d in derivatives]
    derivatives_3d += [np.array([z, z, legendre_1d], dtype=np.float32) for legendre_1d in derivatives]

    return functions, (np.trace(derivatives_3d, axis1=1, axis2=2), [nabla_hat(u_x) for u_x in derivatives_3d])


def find_3tuples(n):
    """Finde alle 3-Tupel (a, b, c), sodass a+b+c=n"""
    result = []

    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - (a + b)
            result.append((a, b, c))

    return result
