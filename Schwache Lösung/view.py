import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_neural_network(model, grid_density):
    """PLottet ein Neuronales Netz über ein Gitter"""
    # Gitterpunkte erzeugen
    x_values = np.linspace(0, 1, grid_density)
    y_values = np.linspace(0, 1, grid_density)
    z_values = np.linspace(0, 1, grid_density)
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)  # Gitter erzeugen
    points = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T  # In die Form (N, 3) bringen

    output = model.predict(points)
    output = output.reshape(X.shape + (3,))  # Zurück in die Form (n, n, n, 3) zum plotten

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Quiverplot für 3D-Visualisierung
    ax.quiver(X, Y, Z, output[..., 0], output[..., 1], output[..., 2], length=0.1, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vektorfeld der Verschiebungen')


def plot_loss(hist):
    """log(Loss) über die Trainingsschritte"""
    indices = list(range(len(hist)))

    plt.figure(figsize=(10, 6))
    plt.plot(indices, np.log10(hist), marker='o')

    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('log10(Loss)')

    plt.grid(True)
