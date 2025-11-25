"""
Unit tests for the Floquet function implementation.

Tests the Julia floquet() function to ensure it produces correct
intermediate matrices and reasonable physical results.
"""

using Test
using LinearAlgebra

# Import Julia implementation
include("../src/floquet_rates.jl")

@testset verbose = true "Floquet Implementation Tests" begin
    # Test parameters
    σz = [1 0; 0 -1]
    σx = [0 1; 1 0]
    ω = 1.5
    ϵ = 0.2
    n_ph = 13
    meas_vec = [1, 0]

    @testset "Basic functionality" begin
        Δ = -6.0
        V = 0.2
        H_0 = Δ / 2 * σz + ϵ * σx
        H_int = V * σz / 2

        prob = floquet(H_0, H_int, ω, n_ph, meas_vec)

        # Probability should be between 0 and 1
        @test 0 <= prob <= 1

        # For these parameters, we expect a non-trivial probability
        @test prob > 0
    end

    @testset "Hamiltonian construction" begin
        Δ = 0.0
        V = 0.2
        H_0 = Δ / 2 * σz + ϵ * σx
        H_int_param = V * σz / 2
        dim = size(H_0, 1)
        n_max = floor(Int, n_ph / 2)

        # Test individual Hamiltonian components
        H_atom = kron(I(n_ph), H_0)
        H_ph = ω * kron(Diagonal(-n_max:n_max), I(dim))
        H_int = kron(offdiagonal_ones(n_ph), H_int_param)

        # Check dimensions
        expected_dim = n_ph * dim
        @test size(H_atom) == (expected_dim, expected_dim)
        @test size(H_ph) == (expected_dim, expected_dim)
        @test size(H_int) == (expected_dim, expected_dim)

        # Check Hermiticity
        H = H_atom + H_ph + H_int
        @test ishermitian(H)
    end

    @testset "Photon state construction" begin
        n_max = floor(Int, n_ph / 2)

        # Ground state should have photon at center
        ψ_photon = onehot(n_ph, n_max + 1)
        @test sum(ψ_photon) == 1
        @test ψ_photon[n_max+1] == 1

        # Test measurement states
        for k in 2:n_ph
            ψ_k = onehot(n_ph, k)
            @test sum(ψ_k) == 1
            @test ψ_k[k] == 1
        end
    end

    @testset "Parameter sensitivity" begin
        # Test different detunings
        detunings = [-6.0, 0.0, 3.0]
        V = 0.2

        results = Float64[]
        for Δ in detunings
            H_0 = Δ / 2 * σz + ϵ * σx
            H_int = V * σz / 2
            prob = floquet(H_0, H_int, ω, n_ph, meas_vec)
            push!(results, prob)

            @test 0 <= prob <= 1
        end

        # Results should be different for different detunings
        @test !all(x -> x ≈ results[1], results)
    end

    @testset "Coupling strength scaling" begin
        Δ = -3.0  # Use non-zero detuning for better sensitivity
        H_0 = Δ / 2 * σz + ϵ * σx

        # Test different coupling strengths
        couplings = [0.05, 0.2, 1.0]
        results = Float64[]

        for V in couplings
            H_int = V * σz / 2
            prob = floquet(H_0, H_int, ω, n_ph, meas_vec)
            push!(results, prob)

            @test 0 <= prob <= 1
        end

        # Results should vary with coupling strength (at least not all identical)
        @test length(unique(results)) > 1
    end

    @testset "Normalization" begin
        Δ = -6.0
        V = 0.2
        H_0 = Δ / 2 * σz + ϵ * σx
        H_int = V * σz / 2
        n_max = floor(Int, n_ph / 2)

        vecs_0 = eigvecs(H_0)
        ψ_g = kron(onehot(n_ph, n_max + 1), vecs_0[:, 1])

        # Ground state should be normalized
        @test ψ_g' * ψ_g ≈ 1.0

        # Measurement states should be normalized
        for k in 2:min(5, n_ph)
            ψ_m = kron(onehot(n_ph, k), meas_vec)
            @test ψ_m' * ψ_m ≈ 1.0
        end
    end
end;
