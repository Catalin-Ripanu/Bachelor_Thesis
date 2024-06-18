import numpy as np

# Import Qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit_aer import AerError

shots = 1e3

circ = QuantumCircuit(3)
circ.h(0)
circ.h(1)
circ.cx(0, 2)
circ.measure_all()

try:
    gpu_simulator = AerSimulator(method='tensor_network', device='GPU')
except AerError as e:
    print(e)

job_automatic = gpu_simulator.run(circ, shots=shots)
counts_automatic = job_automatic.result().get_counts(0)

print(counts_automatic)

