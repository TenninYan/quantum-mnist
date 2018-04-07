"""
test program
creates GHZ state with 10 qubits and sample 100 times
"""

from pyquil.quil import Program
from pyquil.api import QVMConnection, Job
from pyquil.gates import *
import pyquil.paulis as paulis

N_qubit = 10
N_sample = 100
qvm = QVMConnection()

p = Program(H(0)) 
for i in range(N_qubit-1):
	p += Program(CNOT(i,i+1))

print(qvm.wavefunction(p))

print(p)

classical_reg = [i for i in range(N_qubit)]
for i in range(N_qubit):
	p.measure(i,i)

result = qvm.run(p,classical_reg,trials=100)

for r in result:
	print(r)

