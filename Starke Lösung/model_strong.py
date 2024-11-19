import tensorflow as tf
import keras as kr
from help_functions import *


def fun_r(mu, lambd, nabla_hat_u, nabla_hat_v, nabla_u, nabla_v):
    return 2*mu*colon(nabla_hat_u, nabla_hat_v) + lambd*nabla_u*nabla_v


def calculate_right_side(test_functions, g_size, points):

    right_sides = []
    for i in range(len(test_functions[0])):
        right_sides.append(sum(np.dot([0., 1., 0.], test_functions[j][i]) for j in range(len(points)))
                           /len(points) * g_size)

    return right_sides


def init_model(num_layers=100, num_neurons_per_layer=20):

    initializer = kr.initializers.GlorotUniform()
    input_layer = kr.layers.Input(shape=(3,))
    z_input = input_layer[:, 2:]

    # Die versteckten Layer
    previous_layer = input_layer

    for _ in range(num_layers):
        previous_layer = kr.layers.Dense(num_neurons_per_layer, activation='relu', kernel_initializer=initializer,
                                         bias_initializer=initializer)(previous_layer)

    output = kr.layers.Dense(3, activation='linear', kernel_initializer=initializer,
                             bias_initializer=initializer)(previous_layer)

    # Output des sequentiellen Teils wird mit der Distanzfunktion multipliziert
    final_output = kr.layers.Multiply()([output, z_input])
    model = kr.models.Model(inputs=input_layer, outputs=final_output)

    return model


def get_derivatives(function, points, form, mu=1, lambd=1):

    with tf.GradientTape(persistent=True) as tape:

        tape.watch(points)
        u = function(points)

        # Berechne die Jacobi-Matrizen, für jeden Punkt eine, jede 3x3
        u_x = tape.batch_jacobian(u, points)
        nabla_u = tf.linalg.trace(u_x)
        nabla_hat_u = tf.map_fn(nabla_hat, u_x)

        nabla_u_expanded = tf.reshape(nabla_u, [-1, 1, 1])
        identity_matrix = tf.eye(3, dtype=tf.float32)
        inner_strong = lambd*nabla_u_expanded*identity_matrix + 2.0*mu*nabla_hat_u

    inner_strong_x = tape.batch_jacobian(inner_strong, points, experimental_use_pfor=False)
    res = tf.linalg.trace(inner_strong_x)

    del tape

    if form == "weak":
        return nabla_u, nabla_hat_u
    elif form == "strong":
        return res
    elif form == "boundary":
        return inner_strong
    else: 
        return u_x


def compute_loss_strong(model, inner_points, boundary_points, normals, g_values, mu, lambd):
    # Loss im Inneren
    res = get_derivatives(model, inner_points, "strong", mu, lambd)
    loss = tf.reduce_mean(tf.square(res))

    # Werte von N_b2 an den Randpunkten
    res_boundary = get_derivatives(model, boundary_points, "boundary", mu, lambd)
    # N_b2-Werte werden mit den Normalen multipliziert und die Differenz zu g gebildet.
    total_res_boundary = tf.matmul(res_boundary, tf.expand_dims(normals, axis=-1)) #TODO
    total_res_boundary = tf.squeeze(total_res_boundary, axis=-1) - g_values # TODO
    loss += 5*tf.reduce_mean(tf.square(total_res_boundary))

    return loss


def get_grad(model, inner_points, boundary_points, normals, g_values, mu, lambd, t0):
    with tf.GradientTape(persistent=True) as tape:
        loss = compute_loss_strong(model, inner_points, boundary_points, normals, g_values, mu, lambd)

    grad_theta = tape.gradient(loss, model.trainable_variables)
    del tape

    return loss, grad_theta

# Tensorflow-Funktion um die Ausführung zu beschleunigen
@tf.function
def train_step(model, inner_points, boundary_points, normals, g_values, mu, lambd, optim, t0):
    loss, grad_theta = get_grad(model, inner_points, boundary_points, normals, g_values, mu, lambd, t0)

    optim.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss, grad_theta
