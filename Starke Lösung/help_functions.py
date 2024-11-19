import random as rd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.special as sp
from functools import reduce
import operator


def colon(a, b):
    return tf.reduce_sum(tf.multiply(a, b))


def nabla_hat(u_x):
    return 0.5*tf.math.add(u_x, tf.transpose(u_x))


def project_point(point, lb, ub):
    side = rd.randint(1, 5)
    match side:
        case 1:
            point[1] = lb
            normal = [0., -1., 0.]
        case 2:
            point[0] = lb
            normal = [-1., 0., 0.]
        case 3:
            point[1] = ub
            normal = [0., 1., 0.]
        case 4:
            point[0] = ub
            normal = [1., 0., 0.]
        case 5:
            point[2] = ub
            normal = [0., 0., 1.]

    return point, normal


def sample_boundary(n, lb, ub):
    # Ein Punkt im Inneren des Würfels
    points = np.random.rand(n, 3)
    # Der Punkt wird zufällig auf eine der 5 Neumann Seiten projiziert
    result = list(map(lambda x: project_point(x, lb, ub), points))
    points = [result[i][0] for i in range(len(result))]
    normals = [result[i][1] for i in range(len(result))]
    return points, normals

def create_boundary_function(side, lower, upper):
    """
    Returns a function that acts perpendicularly inward on a square on the cube's boundary.
    side: the side on which g acts
    lower: the lower-left corner of the square
    upper: the upper-right corner of the square
    """

    # Derr innere Normalenvektor
    match side:
        case "front":
            inner_vector = tf.constant([0., 1., 0.], dtype=tf.float32)
        case "left":
            inner_vector = tf.constant([1., 0., 0.], dtype=tf.float32)
        case "back":
            inner_vector = tf.constant([0., -1., 0.], dtype=tf.float32)
        case "right":
            inner_vector = tf.constant([-1., 0., 0.], dtype=tf.float32)
        case "top":
            inner_vector = tf.constant([0., 0., -1.], dtype=tf.float32)
        case _:
            raise ValueError(f"Invalid side: {side}")

    def g(x):
        # Obere und untere Grenze als Tensoren
        lower_tf = tf.constant(lower, dtype=tf.float32)
        upper_tf = tf.constant(upper, dtype=tf.float32)

        # Überprüfe, ob x in der Fläche ist, auf die g wirkt
        in_bounds = tf.reduce_all(tf.logical_and(lower_tf <= x, x <= upper_tf))

        # Gib den inneren Vektor zurück wenn ja, sonst 0
        return tf.where(in_bounds, inner_vector, tf.constant([0, 0, 0], dtype=tf.float32))

    return g


def plot_3d_to_1d(function):
    x_values = np.linspace(0, 3, 100)
    x_values = [[x_value, 0, 0] for x_value in x_values]
    y_values = [function(x) for x in x_values]
    plt.plot(x_values, y_values, label='f(x) = sin(x)', color='blue')
    plt.show()


if __name__ == "__main__":
    # x, y = sample_boundary(100, 0., 1.)
    # print(x)
    # print(y)
    # lower_left = [0.2, 0., 0.2]
    # upper_right = [0.8, 0., 0.8]
    # side = "front"
    # g = create_boundary_function(side, lower_left, upper_right)
    # x = tf.Variable(x, dtype="float32")
    # z = tf.map_fn(g, x)
    # print(z)
    print(nabla_hat(tf.convert_to_tensor([[1,2,3], [0,0,0], [0,0,0]], dtype="float32")))