import tensorflow as tf
import keras as kr
from time import time
import matplotlib.pyplot as plt

import model as md
import view
from help_functions import *


if __name__ == "__main__":
    t0 = time()
    DTYPE = 'float32'
    kr.backend.set_floatx(DTYPE)

    # Dimension der Testfunktionsbasis (3d-Legendre-Polynome bis Grad N_test)
    N_test = 5

    # Anzahl der Rand/Innenpunkte
    N_b = 20
    N_r = 30

    xmin = 0.
    xmax = 1.

    np.random.seed(100)

    # Innere Datenpunkte für linke Seite der PDE
    inner_points = [np.random.uniform(xmin, xmax, 3) for _ in range(N_r)]

    # Die Randfunktion g.
    # Die Ecken des Quadrats, auf das g wirken soll
    lower_left = [0.5, 0., 0.5]
    upper_right = [0.7, 0., 0.7]
    side = "front"

    # Randdaten für rechte Seite der PDE
    boundary_points = np.random.uniform(lower_left, upper_right, (N_b, 3))

    # Funktionen um die PDE zu testen und deren Ableitungen
    test_functions, test_derivatives = ([legendre_base(N_test, x)[0] for x in boundary_points],
                                        [legendre_base(N_test, x)[1] for x in inner_points])

    g_size = upper_right[0] - lower_left[0] * upper_right[2] - lower_left[2]

    # Die Lamé-Parameter
    mu = 6.6892*(10**0)
    lambd = 6.6211*(10**2)

    # Initialisiere Modell, also u_\theta
    model = md.init_model(num_layers=8, num_neurons_per_layer=20)
    lr = kr.optimizers.schedules.PiecewiseConstantDecay([100, 500], [1e-2, 1e-2, 1e-3])
    optim = kr.optimizers.Adam(learning_rate=lr)

    # Berechne die rechten Seiten
    right_sides = md.calculate_right_side(test_functions, g_size, boundary_points)

    N = 1000
    hist = []
    inner_points_var = tf.Variable(inner_points, dtype=tf.float32)

    for i in range(N + 1):
        print(i)

        loss, grad_theta = md.train_step(model, inner_points_var, test_derivatives, test_functions, mu, lambd, right_sides, optim, t0)

        hist.append(loss.numpy())

        # Output jede 50 Schritte
        if i % 50 == 0:
            print(time() - t0)
            print('It {:05d}: loss = {:10.8e}'.format(i, loss))

    print('\nComputation time: {} seconds'.format(time() - t0))
    print(model(tf.convert_to_tensor([[0.3, 0.4, 0.3]])))
    print(model(tf.convert_to_tensor([[0.6, 0.1, 0.6]])))
    view.plot_neural_network(model, 10)
    # Erste Iteration wird nicht geplottet
    view.plot_loss(hist[1:])

    plt.tight_layout()
    plt.show()
