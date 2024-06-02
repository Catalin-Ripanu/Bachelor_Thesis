import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

a, b = sympy.symbols('a b')

import cirq_ionq as ionq

# API key is assumed to be stored as an env var named IONQ_API_KEY
service = ionq.Service()

# Parameters that the classical NN will feed values into.
control_params = sympy.symbols('theta_1 theta_2 theta_3')

# Create the parameterized circuit.
qubit = cirq.LineQubit.range(1)[0]
model_circuit = cirq.Circuit(
    cirq.rz(control_params[0])(qubit),
    cirq.ry(control_params[1])(qubit),
    cirq.rx(control_params[2])(qubit))

controller = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='elu'),
    tf.keras.layers.Dense(3)
])

# This input is the simulated mis-calibration that the model will learn to correct.
circuits_input = tf.keras.Input(shape=(),
                # The circuit-tensor has dtype `tf.string`
                dtype=tf.string,
                name='circuits_input')

# Commands will be either `0` or `1`, specifying the state to set the qubit to.
commands_input =    tf.keras.Input(shape=(1,),
                    dtype=tf.dtypes.float32,
                    name='commands_input')

dense_2 = controller(commands_input)

# TFQ layer for classically controlled circuits.
expectation_layer = tfq.layers.ControlledPQC(model_circuit,
                    backend=service.sampler('simulator'),
                    repetitions=3000,
                    # Observe Z
                    operators = cirq.Z(qubit))
expectation = expectation_layer([circuits_input, dense_2])

model = tf.keras.Model(inputs=[circuits_input, commands_input],
    outputs=expectation)
tf.keras.utils.plot_model(model, show_shapes=True, dpi=70)
commands = np.array([[0], [1]], dtype=np.float32)
expected_outputs = np.array([[1], [-1]], dtype=np.float32)
random_rotations = np.random.uniform(0, 2 * np.pi, 3)
noisy_preparation = cirq.Circuit(
    cirq.rx(random_rotations[0])(qubit),
    cirq.ry(random_rotations[1])(qubit),
    cirq.rz(random_rotations[2])(qubit)
)
datapoint_circuits = tfq.convert_to_tensor([
    noisy_preparation
] * 2)  # Make two copies of this circuit

print("Fitting with tfq... this may take some time...")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)
loss = tf.keras.losses.MeanSquaredError()
model.compile(optimizer=optimizer, loss=loss)
history = model.fit(x=[datapoint_circuits, commands],
        y=expected_outputs,
        epochs=30,
        verbose=0)
print ("Plotting now")

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title("Learning to Control a Qubit")
plt.xlabel("Iterations")
plt.ylabel("Error in Control")
plt.show()
