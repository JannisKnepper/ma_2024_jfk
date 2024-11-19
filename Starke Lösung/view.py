import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def plot_neural_network(model, grid_density):
    """
    Plottet ein 3D-3D neuronales Netz
    model: das neuronale Netz als Tensorflow-Modell
    grid_density: Die Dichte des Gitters, über das geplottet wird
    """

    # Generiere ein Gitter
    x_values = np.linspace(0, 1, grid_density)
    y_values = np.linspace(0, 1, grid_density)
    z_values = np.linspace(0, 1, grid_density)

    # Bringe das Gitter in eine passende Form für das Modell
    X, Y, Z = np.meshgrid(x_values, y_values, z_values)
    points = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T

    # Anwenden des Modells auf das Gitter
    output = model.predict(points)
    output = output.reshape(X.shape + (3,))

    U = output[..., 0]
    V = output[..., 1]
    W = output[..., 2]

    # Berechne das Maximum der Beträge zum skalieren
    magnitudes = np.sqrt(U ** 2 + V ** 2 + W ** 2)
    max_magnitude = magnitudes.max()

    # Normalisieren den Output
    U_normalized = U / max_magnitude
    V_normalized = V / max_magnitude
    W_normalized = W / max_magnitude

    # Erstelle den Quiverplot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U_normalized, V_normalized, W_normalized, length=0.1, normalize=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Vektorfeld der Verschiebungen')


def plot_loss(hist):
    """PLottet den log des Loss über die Iterationsschritte"""
    indices = list(range(len(hist)))

    plt.figure(figsize=(10, 6))
    plt.plot(indices, np.log10(hist), marker='o')

    plt.title('Loss vs. Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('log10(Loss)')

    plt.grid(True)
