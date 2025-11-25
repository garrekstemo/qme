"""
Tests for Pauli Master Equation implementation.
"""

using Test
using LinearAlgebra
using Random

include("../src/pauli.jl")
include("../src/bloch_redfield_tensor.jl") # For S() function

# Constants
const fs = 1 / (2.4189e-2)
const eV = 1 / (fs * 0.6582)

@testset "Pauli Master Equation Tests" begin
    N = 3

    # Simple Hamiltonian (diagonal)
    vals = [0.0, 1.0, 2.0]
    kets = Matrix{Float64}(I, N, N)

    # Coupling: transitions 1<->2 and 2<->3
    # Operator connecting 1 and 2
    A12 = zeros(N, N)
    A12[1, 2] = 1.0
    A12[2, 1] = 1.0

    # Simple constant NPS
    NPS_const(ω) = 0.1

    A_operators = [(A12, NPS_const)]

    W = construct_W(A_operators, vals, kets, N)

    @test size(W) == (N, N)

    # Check conservation of probability (columns sum to zero)
    # dP_a/dt = sum_b W_{ab} P_b
    # sum_a dP_a/dt = sum_a sum_b W_{ab} P_b = sum_b P_b (sum_a W_{ab})
    # For trace preservation, sum_a W_{ab} must be 0 for all b

    for b in 1:N
        col_sum = sum(W[:, b])
        @test abs(col_sum) < 1e-10
    end

    # Check specific rates
    # Transition 1->2: rate should be |<1|A|2>|^2 * NPS(E1-E2) = 1 * 0.1 = 0.1
    # W[2, 1] is rate 1 -> 2
    @test W[2, 1] ≈ 0.1

    # Transition 2->1: rate should be |<2|A|1>|^2 * NPS(E2-E1) = 1 * 0.1 = 0.1
    # W[1, 2] is rate 2 -> 1
    @test W[1, 2] ≈ 0.1

    # Diagonal elements (outflow)
    # W[1, 1] = - (rate 1->2 + rate 1->3 + ...) = -0.1
    @test W[1, 1] ≈ -0.1
    # W[2, 2] = - (rate 2->1 + rate 2->3) = -0.1 (since 2->3 is zero)
    @test W[2, 2] ≈ -0.1

    # State 3 is decoupled
    @test W[3, 3] == 0.0
    @test W[1, 3] == 0.0
    @test W[3, 1] == 0.0
end;