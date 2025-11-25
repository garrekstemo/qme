"""
Tests for random quantum network generation.
"""

using Test
using LinearAlgebra
using Random
using StatsBase

# Constants
const fs = 1 / (2.4189e-2)
const eV = 1 / (fs * 0.6582)

@testset "Random Quantum Network Generation" begin
    N = 10
    σ_E = 100e-3 * eV
    σ_V = 50e-3 * eV

    Random.seed!(42)

    # Generate energies
    Es = σ_E * randn(N)
    @test length(Es) == N

    # Generate couplings
    all_pairs = [(j, k) for j in 1:N for k in j+1:N]
    @test length(all_pairs) == N * (N - 1) / 2

    indices = sample(1:length(all_pairs), N, replace=false)
    pairs = all_pairs[indices]
    @test length(pairs) == N

    Vs = [(i, j, σ_V * randn()) for (i, j) in pairs]

    # Construct Hamiltonian
    H = Matrix(Diagonal(Es))
    for (j, k, v) in Vs
        H[j, k] = v
        H[k, j] = conj(v)
    end

    @test size(H) == (N, N)
    @test ishermitian(H)

    # Check sparsity structure (should have N diagonal + 2*N off-diagonal non-zeros)
    # Note: Random values are extremely unlikely to be exactly zero
    n_nonzero = count(x -> abs(x) > 0, H)
    @test n_nonzero == N + 2 * N
end;
