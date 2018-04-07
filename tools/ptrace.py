import itertools



def B(N):
    return itertools.product(*(N*[[0, 1]]))

def repr_to_int(base, _repr):
    return sum([
        _repr[k] * base**k
        for k in range(len(_repr))
    ])

def ptrace(n, target_qubit_indices, probabilities):
    '''
    n: number of qubits
    target_qubit_indices: list of indices left over after trace
    probabilities: list of probabilities for all states,
    (should be of size 2^n)
    '''

    traced_out_indices = [
        j
        for j in range(n)
        if j not in target_qubit_indices
    ]

    output_totals = {}
    for target_bs in B(len(target_qubit_indices)):
        total = 0
        full_bs = n * [0]
        for i, val in zip(target_qubit_indices, target_bs):
            full_bs[i] = val
        for trace_bs in B(len(traced_out_indices)):
            for j, val in zip(traced_out_indices, trace_bs):
                full_bs[j] = val
            total += probabilities[repr_to_int(2, full_bs)]
        output_totals[target_bs] = total
    
    return [
        output_totals[bs]
        for bs in B(len(target_qubit_indices))
    ]
    

if __name__ == '__main__':
    print(ptrace(
        2,
        [0],
        [1/4., 1/4., 1/4., 1/4.]
    ))

    print(ptrace(
        2,
        [0, 1],
        [1/4., 1/4., 1/4., 1/4.]
    ))
    
    print(ptrace(
        2,
        [],
        [1/4., 1/4., 1/4., 1/4.]
    ))