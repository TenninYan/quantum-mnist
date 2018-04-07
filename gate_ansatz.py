"""
gate ansatz program
chaining 2 qubit arbitrary gates
"""

from pyquil.quil import Program
from pyquil.gates import *
import pyquil.paulis as paulis
from numpy import pi, linspace
from numpy.random import rand

N_qubit = 16
N_unit_theta = 15

def input_gates(data):
    """
    returns Program object representing input gates
    """
    p = Program() #initial gate
    for i in range(N_qubit):
        p += Program(RY(pi/2*data[i%len(data)])(i))

    return p

def unit_gate(theta, target = (0,1)):
    """
    returns Program object arbitrary 2 qubit gate (15 params)
    """
    p = Program()
    
    #1 qubit rotation on each qubit
    p += arbitrary1qubitrotation(theta[0:3],target[0])
    p += arbitrary1qubitrotation(theta[3:6],target[1])
       
    # 2 qubit canonical gate
    p += Program(CNOT(target[0],target[1]))
    p += Program(RZ(theta[6])(target[1]))
    p += Program(RX(theta[7])(target[1]))
    p += Program(RZ(theta[8])(target[1]))
    p += Program(CNOT(target[0],target[1]))
    
    #1 qubit rotation on each qubit
    p += arbitrary1qubitrotation(theta[9:12],target[0])
    p += arbitrary1qubitrotation(theta[12:15],target[1])
    return p
    
def arbitrary1qubitrotation(theta, target):
    p = Program(RZ(theta[0])(target))
    p += Program(RX(theta[1])(target))
    p += Program(RZ(theta[2])(target))
    return p
    

def output_gates(theta, depth = N_qubit/2):
    """
    returns U(theta)
    """
    p = Program()
    N_theta_used = 0
    for d in range(depth):
        if d%2 == 0:
            for i in range(N_qubit//2):
                p += unit_gate(theta[N_theta_used:N_theta_used+N_unit_theta], target = (i*2, 2*i+1))
                N_theta_used += N_unit_theta
        
        elif d%2 == 1:
            for i in range(N_qubit//2 -1):
                p += unit_gate(theta[N_theta_used:N_theta_used+N_unit_theta], target = (2*i+1, 2*i+2))
                N_theta_used += N_unit_theta
    return p

def get_init_theta(depth):
    N_theta_used = 0
    for d in range(depth):
        if d%2 == 0:
            for i in range(N_qubit//2):
                N_theta_used += N_unit_theta
        
        elif d%2 == 1:
            for i in range(N_qubit//2 -1):
                N_theta_used += N_unit_theta

    return [pi*rand() for k in range(N_theta_used)]


if __name__ == '__main__':
    depth = 8
    theta_init = get_init_theta(depth)
    p = input_gates(linspace(0,1,N_qubit)) 
    p2 = output_gates(theta_init, depth)
    print(p)
    print(p2)