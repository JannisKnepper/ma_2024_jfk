import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# def plot_3d_vector_field(model, grid_density):
#     # Define a grid of points where we will evaluate the black-box function
#     x = np.linspace(0, 1, grid_density)
#     y = np.linspace(0, 1, grid_density)
#     z = np.linspace(0, 1, grid_density)
#
#     # Create a 3D meshgrid for X, Y, and Z values
#     X, Y, Z = np.meshgrid(x, y, z)
#
#     # Prepare arrays to store the vector field components
#     U = np.zeros_like(X)  # To hold the X-component of the resulting vectors
#     V = np.zeros_like(Y)  # To hold the Y-component of the resulting vectors
#     W = np.zeros_like(Z)  # To hold the Z-component of the resulting vectors
#
#     # Evaluate the black box function at each grid point
#     for i in range(X.shape[0]):
#         for j in range(X.shape[1]):
#             for k in range(X.shape[2]):
#                 # Input vector at grid point (X[i, j, k], Y[i, j, k], Z[i, j, k])
#                 input_vector = [X[i, j, k], Y[i, j, k], Z[i, j, k]]
#
#                 # Get the output vector from the black box function
#                 output_vector = model(tf.convert_to_tensor([input_vector], dtype=tf.float32)).numpy()
#
#                 # Store the output vector components
#                 U[i, j, k] = output_vector[0]  # X-component of output
#                 V[i, j, k] = output_vector[1]  # Y-component of output
#                 W[i, j, k] = output_vector[2]  # Z-component of output
#
#     # Plot the 3D vector field using quiver
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot the vectors
#     ax.quiver(X, Y, Z, U, V, W, length=0.5, normalize=True)
#
#     # Set limits for the plot
#     ax.set_xlim([-0.5, 1.5])
#     ax.set_ylim([-0.5, 1.5])
#     ax.set_zlim([-0.5, 1.5])
#
#     # Set labels and title
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Z-axis')
#     ax.set_title('Vektorfeld')


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


if __name__ == "__main__":
    def model(input_vector):
        x, y, z = input_vector
        return [y, 2 * x, z]

    plot_3d_vector_field(model, 5)
