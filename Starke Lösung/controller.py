import numpy as np
import tensorflow as tf
import keras as kr
from time import time
import matplotlib.pyplot as plt

import model_strong as md_strong
import view
from help_functions import *


if __name__ == "__main__":
    t0 = time()
    DTYPE = 'float32'
    kr.backend.set_floatx(DTYPE)

    # Anzahl der Rand-/Innenpunkte
    N_b = 500
    N_r = 2000

    xmin = 0.
    xmax = 1.

    # numpy-Seed
    np.random.seed(100)

    # Innere Datenpunkte für linke Seite der PDE
    inner_points = [np.random.uniform(xmin, xmax, 3) for _ in range(N_r)]
    inner_points = tf.Variable(inner_points, dtype=tf.float32)

    # Die Randfunktion g.
    # Die Ecken des Quadrats, auf das g wirken soll
    lower_left = [0.5, 0., 0.5]
    upper_right = [0.9, 0., 0.9]
    side = "front"

    # Randdaten für rechte Seite der PDE
    boundary_points, normals = sample_boundary(N_b, xmin, xmax)
    boundary_points = tf.Variable(boundary_points, dtype=DTYPE)
    normals = tf.Variable(normals, dtype=DTYPE)

    g = create_boundary_function(side, lower_left, upper_right)
    g_values = tf.map_fn(g, boundary_points)

    # Die Lamé-Parameter (Haut, vgl. Shape reconstruction in linear elasticity)
    mu = 6.6892*(10**3)
    lambd = 6.6211*(10**5)

    # Initialisiere Modell, also u_\theta
    model = md_strong.init_model(num_layers=8, num_neurons_per_layer=20)
    lr = kr.optimizers.schedules.PiecewiseConstantDecay([600, 900], [1e-2, 1e-3, 1e-4])
    optim = kr.optimizers.Adam(learning_rate=lr)

    N = 1000
    hist = []

    for i in range(N + 1):
        print(i)

        loss, grad_theta = md_strong.train_step(model, inner_points, boundary_points, normals, g_values, mu, lambd, optim, t0)

        hist.append(loss.numpy())

        # Output jede 50 Schritte
        if i % 50 == 0:
            print(time() - t0)
            print('It {:05d}: loss = {:10.8e}'.format(i, loss))

    print('\nComputation time: {} seconds'.format(time() - t0))
    print(model(tf.convert_to_tensor([[0.5, 0., 0.5]])))
    print(model(tf.convert_to_tensor([[0.8, 0.8, 0.8]])))
    view.plot_neural_network(model, 10)
    
    # Erste Iteration wird nicht geplottet
    view.plot_loss(hist[1:])

    plt.tight_layout() 
    plt.show()
