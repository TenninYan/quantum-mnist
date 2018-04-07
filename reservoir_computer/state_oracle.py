'''
In this model, the reservoir is treated as
an oracle by the hybrid classical controller.

(1) The quantum state is initialized using gates determined by
rotation parameters from the continuous input data sample
(2) A quantum circuit with high connectivity is generated with random
(seeded) rotational parameters, and provides the reservoir dynamics
(3) Measurement is made in a static basis
'''


from pyquil.quil import Program
from pyquil.gates import *
from pyquil.api import QVMConnection, Job
import pyquil.paulis as paulis
import numpy as np
from numpy import pi, linspace
from numpy.random import rand
import pickle as pickle

N_qubit = 16
N_unit_theta = 15

def generate_input_gates(data):
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
    p += Program(RX(theta[6])(target[1]))
    p += Program(RY(theta[7])(target[1]))
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


def load_mnist_data(
    train_filename='../train_pca.pickle',
    test_filename='../test_pca.pickle'
):
    with open(train_filename, 'rb') as infile:
        train_data = pickle.load(infile)

    with open(test_filename, 'rb') as infile:
        test_data = pickle.load(infile)
    
    return train_data, test_data


def write_reservoir_output(
    train, test,
    train_filename='../train_pca_reservoir_output.pickle',
    test_filename='../test_pca_reservoir_output.pickle'
):
    with open(train_filename, 'wb') as outfile:
        pickle.dump(train, outfile)

    with open(test_filename, 'wb') as outfile:
        pickle.dump(test, outfile)

def reservoir_compute(sample, reservoir_circuit, qcircuit_callback):
    '''
    sample: numpy array of floating point data
    reservoir_circuit: Program to implement quantum reservoir computer
    qcircuit_callback: a callback to generate the input state
    initialization circuit
    '''

    p = qcircuit_callback(sample)
    psi = qvm.wavefunction(p + p_reservoir)
    probabilities = np.real(psi.amplitudes.conjugate() * psi.amplitudes)
    return probabilities


if __name__ == '__main__':
    qvm = QVMConnection()

    # circuit depth parameter
    depth = 8

    # initialize repeatable random reservoir
    np.random.seed(12939229)
    theta_init = get_init_theta(depth)
    p_reservoir = output_gates(theta_init, depth)


    # load data
    train, test = load_mnist_data()
    num_train_samples = train[0].shape[0]
    num_test_samples = test[0].shape[0]

    # for debugging
    num_train_samples = min(20, num_train_samples)
    num_test_samples = min(20, num_test_samples)


    # run all training & testing samples through reservoir
    from tqdm import tqdm

    train_outputs = []
    print('generating training data reservoir output...')
    for j in tqdm(range(num_train_samples)):
        train_outputs.append(
            reservoir_compute(
                train[0][j],
                p_reservoir,
                generate_input_gates
            )
        )

    test_outputs = []
    print('generating test data reservoir output...')
    for j in tqdm(range(num_test_samples)):
        test_outputs.append(
            reservoir_compute(
                test[0][j],
                p_reservoir,
                generate_input_gates
            )
        )

    # save outputs to disk
    write_reservoir_output(
        [
            np.array(train_outputs),
            train[1] # labels
        ],
        [
            np.array(test_outputs),
            test[1] # labels
        ]
    )


# import sklearn.datasets
# 
# data = sklearn.datasets.make_blobs(
#     n_samples=100,
#     n_features=16,
#     centers=10
# )
# 
# N_qubit = 10
# 
# # Define reservoir oracle as program
# p = Program(H(0))