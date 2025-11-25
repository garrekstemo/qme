using LinearAlgebra

# The onehot(n, k) function is defined in src/functions.jl
include("src/helper_functions.jl")

function floquet(H_0, H_int, ω, n_ph, meas_vec)
    dim = size(H_0, 1)  # dimension of atomic system
    n_max = floor(Int, n_ph / 2)  # photon range

    H_atom = kron(I(n_ph), H_0)  # atomic Hamiltonian
    H_ph = ω * kron(Diagonal(-n_max:n_max), I(dim))  # photon Hamiltonian

    # Construct the interaction Hamiltonian
    H_int = kron(offdiagonal_ones(n_ph), H_int)
    
    H = H_atom + H_int + H_ph  # Put it all together

    # spectral decomposition of H0 and H
    vecs_0 = eigvecs(H_0)
    vecs = eigvecs(H)

    # ground state vector
    ψ_g = onehot(n_ph, n_max + 1)  # all zeros except 1 at center
    ψ_g = kron(ψ_g, vecs_0[:, 1])

    # sum over all transitions
    overlap_probability = 0
    for k_c in 2:n_ph
        k_vec = onehot(n_ph, k_c)
        ψ_m = kron(k_vec, meas_vec)
        for vec in eachcol(vecs)
            overlap_probability += abs2((ψ_m' * vec) * (vec' * ψ_g))
        end
    end
    return overlap_probability
end