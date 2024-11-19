# import random as rd
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
import scipy.special as sp
# from functools import reduce
# import operator


def colon(a, b):
    return tf.reduce_sum(tf.multiply(a, b))


def nabla_hat(u_x):
    return 0.5*tf.math.add(u_x, tf.transpose(u_x))


# def project_point(point, lb, ub):
#     side = rd.randint(1, 5)
#     match side:
#         case 1:
#             point[1] = lb
#         case 2:
#             point[0] = lb
#         case 3:
#             point[1] = ub
#         case 4:
#             point[0] = ub
#         case 5:
#             point[2] = ub
#
#     return point
#
#
# def sample_boundary(n, lb, ub):
#     # Ein Punkt im Inneren des Würfels
#     points = np.random.rand(n, 3)
#     # Der Punkt wird zufällig auf eine der 5 Neumann Seiten projiziert
#     return list(map(lambda x: project_point(x, lb, ub), points))


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


# def legendre_base_tensor(n, x):
#     """
#     Gibt eine Liste der Legendre-Polynome von R^3 zu R^3 bis zum Grad n zurück
#     """
#
#     return tf.py_function(func=lambda x: tf.convert_to_tensor(legendre_base(x[0], x[1])), inp=[n, [x]], Tout=tf.float32)


def find_3tuples(n):
    """Finde alle 3-Tupel (a, b, c), sodass a+b+c=n"""
    result = []

    for a in range(n + 1):
        for b in range(n - a + 1):
            c = n - (a + b)
            result.append((a, b, c))

    return result

# Wird momentan nicht benutzt
# def create_boundary_function(side, lower, upper):
#     """
#     Gibt eine Funktion zurück, die senkrecht nach innen auf ein Quadrat am Würfelrand wirkt
#     side: die Seite, auf die g wirkt
#     lower: die linke untere Ecke des Quadrats
#     upper: die rechte obere Ecke des Quadrats
#     """
#
#     match side:
#         case "front":
#             inner_vector = [0, 1, 0]
#         case "left":
#             inner_vector = [1, 0, 0]
#         case "back":
#             inner_vector = [0, -1, 0]
#         case "right":
#             inner_vector = [-1, 0, 0]
#         case "top":
#             inner_vector = [0, 0, -1]
#
#     def g(x):
#         if lower[0] <= x[0] <= upper[0] and lower[1] <= x[1] <= upper[1] and lower[2] <= x[2] <= upper[2]:
#             return inner_vector
#         else:
#             return [0, 0, 0]
#
#     return g


# def plot_3d_to_1d(function):
#     x_values = np.linspace(0, 3, 100)
#     x_values = [[x_value, 0, 0] for x_value in x_values]
#     y_values = [function(x) for x in x_values]
#     plt.plot(x_values, y_values, label='f(x) = sin(x)', color='blue')
#     plt.show()


if __name__ == "__main__":
    x = np.random.uniform(0,1,10)
    b = legendre_base(1,x)
    print(x)