import tensorflow as tf
import keras as kr
from help_functions import *


def fun_r(mu, lambd, nabla_hat_u, nabla_hat_v, nabla_u, nabla_v):
    """Die linke Seite der schwachen Formulierung der PDE"""
    return 2*mu*colon(nabla_hat_u, nabla_hat_v) + lambd*nabla_u*nabla_v


def calculate_right_side(test_functions, g_size, points):
    """

    :param test_functions: Werte der Testfunktionen an den inneren Punkten.
    Eine Zeile pro Punkt, eine Spalte pro Funktion.
    :param g_size: Die Größe der Fläche, auf die g auf dem Neumann-Rand wirkt.
    :param points: Die Inneren Punkte zur Berechnung des Integrals.
    :return: Der Wert der rechten Seite der PDE für jede der Testfunktionen
    """

    right_sides = []
    for i in range(len(test_functions[0])):
        right_sides.append(sum(np.dot([0., -1., 0.], test_functions[j][i]) for j in range(len(points)))
                           /len(points) * g_size)

    return right_sides


def init_model(num_layers=100, num_neurons_per_layer=20):
    """
    Erzeugt ein Modell mit den angegebenen Parametern. Zusätzlich ein Layer am Ende, der mit dem ursprünglichen z
    multipliziert, um die Abstandsfunktion zu implementieren.
    """

    initializer = kr.initializers.GlorotNormal()
    input_layer = kr.layers.Input(shape=(3,))
    z_input = input_layer[:, 2:]*2

    # Die versteckten Layer
    previous_layer = input_layer

    for _ in range(num_layers):
        previous_layer = kr.layers.Dense(num_neurons_per_layer, activation=tf.nn.relu, kernel_initializer=initializer,
                                         bias_initializer=initializer)(previous_layer)

    output = kr.layers.Dense(3, activation='linear', kernel_initializer=initializer,
                             bias_initializer=initializer)(previous_layer)

    # Output des sequentiellen Teils wird mit der Distanzfunktion multipliziert
    final_output = kr.layers.Multiply()([output, z_input])
    model = kr.models.Model(inputs=input_layer, outputs=final_output)

    return model


def get_derivatives(function, inner_points):
    """Die Ableitungen des neuronalen Netzes an den inneren Punkten zur Berechnung des Residuums"""

    # A tf.GradientTape is used to compute derivatives in TensorFlow
    with tf.GradientTape(persistent=True) as tape:

        tape.watch(inner_points)
        u = function(inner_points)

    # Berechne die Jacobi-Matrizen, für jeden Punkt eine, jede 3x3
    u_x = tape.batch_jacobian(u, inner_points)

    # Nabla * u (x)
    nabla_u = tf.linalg.trace(u_x)

    # Nabla_hut(u)(x)
    nabla_hat_u = tf.map_fn(nabla_hat, u_x)

    del tape

    return nabla_u, nabla_hat_u


def compute_loss(model, inner_points, test_derivatives, test_functions, mu, lambd, right_sides, t0):
    """

    :param model: Das Modell, dessen Loss berechnet werden soll.
    :param inner_points: Die Inneren Punkte zur Berechnung des Integrals.
    :param test_derivatives: Die Werte der Ableitungen der Testfunktionen an den inneren Punkten.
    :param test_functions: Die Werte der Testfunktionen. Nur für die Anzahl der Testfunktionen.
    :param mu: Lamé-Parameter.
    :param lambd: Lamé-Parameter.
    :param right_sides: Die rechte Seite der PDE für jede der Testfunktionen.
    :param t0: Startzeitpunkt um ggf. Laufzeit ausgeben zu können.
    :return: Loss des Modells, berechnet als Integral durch MC an den inneren Punkten
    """

    # Berechne die Ableitungen des Modells
    nabla_u, nabla_hat_u = get_derivatives(model, inner_points)

    total_loss = 0

    # Schleife über alle Testfunktionen
    for i in range(len(test_functions[0])):
        inner_values = [fun_r(mu, lambd, nabla_hat_u[j], test_derivatives[j][1][i],
                              nabla_u[j], test_derivatives[j][0][i]) for j in range(len(inner_points))]  # TODO nicht hardcoden
        left_side = tf.reduce_sum(inner_values)/len(nabla_u)
        right_side = tf.cast(right_sides[i], tf.float32)

        loss = tf.reduce_mean(tf.square(right_side-left_side))
        total_loss += loss

    average_loss = total_loss/len(test_functions)

    return average_loss


def get_grad(model, inner_points, test_derivatives, test_functions, mu, lambd, right_sides, t0):
    """Die Ableitung des Loss nach den Gewichten und Bias"""
    with tf.GradientTape(persistent=True) as tape:
        loss = compute_loss(model, inner_points, test_derivatives, test_functions, mu, lambd, right_sides, t0)

    g = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, g


@tf.function
def train_step(model, inner_points, test_derivatives, test_functions, mu, lambd, right_sides, optim, t0):
    # Loss und dessen Ableitung nach Gewichten und Bias
    loss, grad_theta = get_grad(model, inner_points, test_derivatives, test_functions, mu, lambd, right_sides, t0)

    # Gradient-descent
    optim.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss


if __name__ == "__main__":
    # Tests
    DTYPE = 'float32'

    xmin = 0.
    xmax = 1.
    inner_points = [np.random.uniform(xmin, xmax, 3) for _ in range(4)]
    inner_points_vec = tf.Variable(inner_points)

    #boundary_points = sample_boundary(5, xmin, xmax)

    #test_functions, test_derivatives = ([legendre_base(1, x)[0] for x in boundary_points],
    #                                    [legendre_base(1, x)[1] for x in inner_points])

    md = init_model(2, 3)
    # print(md(tf.convert_to_tensor([[1., 2., 0.]])))
    c = get_derivatives(md, inner_points_vec)
    print(c)
    # b = compute_loss(md, inner_points, test_derivatives, test_functions, 1, 1, [1,1,1,1,1,1,1,1,1,1,1,1])
    # print(b)
    # lower_left = [0., 0, 0]
    # upper_right = [1, 0, 1]
    # # Mögliche Seiten sind front, back, left, right, top
    # g = create_boundary_function("front", lower_left, upper_right)
    # rs = calculate_right_side(test_functions, g, boundary_points)
    # print(rs)
