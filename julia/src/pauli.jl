"""
    construct_W(A_operators, vals, kets, N)

Construct the transition rate matrix W for the Pauli Master Equation.

# Arguments
- `A_operators`: List of tuples (operator, nps_function).
- `vals`: Eigenvalues of the Hamiltonian.
- `kets`: Eigenvectors of the Hamiltonian.
- `N`: System size.
"""
function construct_W(A_operators, vals, kets, N)
    W = zeros(N, N)
    for (A_op, nps) in A_operators
        A = kets' * A_op * kets
        for a in 1:N
            # Population outflow (diagonal elements)
            # Sum over all possible transitions b -> a (absorption/emission)
            # Note: The sign convention here assumes W[a,a] stores the total decay rate out of state a
            # or it's part of the master equation dP_a/dt = sum_b (W_{ab} P_b - W_{ba} P_a)
            # Let's match the implementation from 4.4_pauli_master_equation.jl

            W[a, a] -= sum([A[a, b] * A[b, a] * nps(vals[a] - vals[b]) for b in 1:N])

            for b in 1:N
                # Population inflow from b to a
                W[a, b] += A[b, a] * A[a, b] * nps(vals[b] - vals[a])
            end
        end
    end
    return W
end
