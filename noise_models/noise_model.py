"""
test program with noise model (pauli channel)
creates GHZ state with 10 qubits and sample 100 times
"""

from pyquil.quil import Program
from pyquil.api import QVMConnection, Job
from pyquil.gates import *
import pyquil.paulis as paulis
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


if __name__ == '__main__':
	N_qubit = 10
	for noise_prob in [0., .01, .05, .1]:
		#1% chance of each gate at each timestep
		pauli_channel = [noise_prob] * 3

		noisy_qvm = QVMConnection(gate_noise=pauli_channel)

		p = Program(H(0)) 
		for i in range(N_qubit-1):
			p += Program(CNOT(i,i+1))

		print(noisy_qvm.wavefunction(p))

		print(p)

		classical_reg = [i for i in range(N_qubit)]
		for i in range(N_qubit):
			p.measure(i,i)

		num_trials = 1000
		result = noisy_qvm.run(
			p,
			classical_reg,
			trials=num_trials
		)


		for r in result:
			print(r)

		plt.hist(
			[2 * (sum(trial) - 5) for trial in result],
			label='noise = %s' % noise_prob,
			alpha=.5
		)


	plt.legend()
	plt.title(
		'%s trials of 10 Qubit GHZ state $\langle S_z \\rangle$, with QVM noise model' % num_trials
	)
	plt.xlabel('$\langle S_z \\rangle$')
	plt.ylabel('#')
	plt.xlim((-10, 10))
	plt.savefig('../figures/NoiseModelPauliChannel_10qubitGHZstate1000TrialMeasurements.png', dpi=300)
