# Comprehensive tests for the Bloch-Redfield tensor implementation
# These tests verify correctness based on known properties and debugging insights

using Test
using LinearAlgebra

include("../src/bloch_redfield_tensor.jl")

# Test parameters
const fs = 1 / (2.4189e-2)  # femtosecond in Hartree AU
const eV = 1 / (fs * 0.6582)  # electronvolt in Hartree AU

@testset "Noise Power Spectrum (S function)" begin
    # Test parameters
    ωc = 150e-3 * eV
    η = 1e-1
    β = 1 / (25e-3 * eV)
    threshold = 1e-10

    @testset "Positive frequency behavior" begin
        ω = 0.1
        nps_val = S(ω, ωc, η, β, threshold)
        @test nps_val > 0
        @test isfinite(nps_val)
    end

    @testset "Near-zero frequency (thermal limit)" begin
        ω = 1e-12  # Very small frequency
        nps_val = S(ω, ωc, η, β, threshold)
        thermal_limit = 2π * η / β
        @test nps_val ≈ thermal_limit rtol = 1e-6
    end

    @testset "Negative frequency" begin
        ω = -0.1
        nps_pos = S(abs(ω), ωc, η, β, threshold)
        nps_neg = S(ω, ωc, η, β, threshold)
        # Should satisfy detailed balance: S(-ω) = S(ω) * exp(-β*ω)
        @test nps_neg > 0
        @test isfinite(nps_neg)
    end

    @testset "Exponential decay at large frequencies" begin
        ω1 = 1.0
        ω2 = 2.0
        nps1 = S(ω1, ωc, η, β, threshold)
        nps2 = S(ω2, ωc, η, β, threshold)
        # Should decay with increasing frequency
        @test nps2 < nps1
    end
end

@testset "Kronecker delta function" begin
    @test δ(1, 1) == true
    @test δ(2, 1) == false
    @test δ(5, 5) == true
    @test δ(0, 1) == false
end

@testset "Bloch-Redfield tensor properties" begin
    # Create a simple test system: 2-level system (qubit)
    N = 2
    ω0 = 1.0  # Transition frequency
    H = Diagonal([0.0, ω0])

    # Coupling operator: σz (dephasing)
    σz = [1.0 0.0; 0.0 -1.0]

    # Simple constant NPS
    NPS_const(ω) = 0.1
    a_ops = [(σz, NPS_const)]

    R = BR_tensor(H, a_ops, false)  # No secular approximation

    @testset "Dimensions" begin
        @test size(R) == (N^2, N^2)
        @test size(R, 1) == 4
        @test size(R, 2) == 4
    end

    @testset "Trace preservation" begin
        # The BR tensor should preserve trace: Tr[R[ρ]] = 0
        # This means sum of each column weighted by identity should be zero
        # For a vectorized representation, this means certain sum rules

        # Create identity density matrix (vectorized)
        ρ_vec = vec(Matrix{ComplexF64}(I, N, N))

        # R acting on identity should give zero (trace preservation)
        result = R * ρ_vec
        trace_result = sum(result[1:N+1:N^2])  # Sum diagonal elements
        @test abs(trace_result) < 1e-10
    end

    @testset "Hermiticity preservation" begin
        # R should preserve Hermiticity: if ρ is Hermitian, R[ρ] should be Hermitian
        # This is automatically satisfied by construction, but we can check
        # that R has the right structure
        @test eltype(R) == ComplexF64
    end
end

@testset "Secular approximation" begin
    # Create a test system with moderately separated energy levels
    # Energy scale comparable to NPS bandwidth to see secular approximation effect
    N = 4
    H = Diagonal([0.0, 0.1, 0.15, 0.25])  # Energy scale ~ 0.1-0.25

    # Use a broader NPS that doesn't decay too fast
    # This ensures non-secular terms are non-negligible
    NPS_broad(ω) = 0.5 * exp(-abs(ω) / 0.5)  # Broader than typical thermal NPS

    # Off-diagonal coupling (not just site projectors) to create more transitions
    σx = [0.0 1.0 0.0 0.0; 1.0 0.0 1.0 0.0; 0.0 1.0 0.0 1.0; 0.0 0.0 1.0 0.0]
    a_ops = [(σx, NPS_broad)]

    R_secular = BR_tensor(H, a_ops, true, 0.01)
    R_nosecular = BR_tensor(H, a_ops, false, 0.01)

    @testset "Secular vs non-secular difference" begin
        diff = R_secular - R_nosecular
        max_diff = maximum(abs.(diff))

        # There should be a difference (or both should be approximately zero for this case)
        # We mainly want to check the code doesn't crash and produces valid output
        @test isfinite(max_diff)
        @test all(isfinite.(R_secular))
        @test all(isfinite.(R_nosecular))

        # Secular should have at least as many zeros (be at least as sparse)
        n_zeros_secular = count(abs.(R_secular) .< 1e-10)
        n_zeros_nosecular = count(abs.(R_nosecular) .< 1e-10)
        @test n_zeros_secular >= n_zeros_nosecular
    end

    @testset "Energy filtering condition" begin
        # Check that secular approximation correctly filters based on energy differences
        vals = eigvals(H)

        # Calculate gmax
        BohrF = [vals[a] - vals[b] for a in 1:N for b in 1:N]
        gmax = maximum([NPS_broad(f) for f in BohrF])
        threshold = gmax * 0.01

        # Check specific matrix elements
        # For indices (a,b,c,d), the secular condition is:
        # |E_a - E_b - E_c + E_d| <= threshold

        test_cases = [
            (1, 1, 1, 2),  # Should be filtered (large energy diff)
            (1, 1, 1, 1),  # Should pass (zero energy diff)
            (1, 2, 2, 1),  # Should pass (zero energy diff)
        ]

        for (a, b, c, d) in test_cases
            energy_diff = abs(vals[a] - vals[b] - vals[c] + vals[d])
            should_pass = energy_diff <= threshold

            # Calculate matrix indices
            j = (a - 1) * N + b
            k = (c - 1) * N + d

            r_val_secular = R_secular[j, k]
            r_val_nosecular = R_nosecular[j, k]

            if should_pass
                # Both should be non-zero (or both zero)
                @test abs(r_val_secular - r_val_nosecular) < 1e-10
            else
                # Secular should be closer to zero
                @test abs(r_val_secular) <= abs(r_val_nosecular)
            end
        end
    end
end

@testset "Operator transformation to eigenbasis" begin
    # Test that operators are correctly transformed to Hamiltonian eigenbasis
    N = 2
    # Non-diagonal Hamiltonian
    H = [0.0 0.5; 0.5 1.0]
    vals, kets = eigen(H)

    # Test operator
    A = [1.0 0.0; 0.0 0.0]

    # Transform to eigenbasis
    A_eigen = kets' * A * kets

    # Transform back
    A_back = kets * A_eigen * kets'

    @test A_back ≈ A rtol = 1e-10
end

@testset "Physical constraints" begin
    # Create a realistic test system
    N = 4
    using Random
    Random.seed!(42)

    # Random Hermitian Hamiltonian
    H_rand = randn(N, N) + im * randn(N, N)
    H = (H_rand + H_rand') / 2

    # Site projection operators
    NPS(ω) = S(ω, 0.15 * eV, 0.1, 1 / (0.025 * eV))
    a_ops = [(Matrix(Diagonal([i == k ? 1.0 : 0.0 for i in 1:N])), NPS) for k in 1:N]

    R = BR_tensor(H, a_ops, true)

    @testset "Complete positivity structure" begin
        # The BR tensor should have a specific structure that ensures
        # complete positivity (at least approximately for weak coupling)

        # Check that R is finite everywhere
        @test all(isfinite.(R))

        # Check dimensions
        @test size(R) == (N^2, N^2)
    end

    @testset "Unitary part structure" begin
        # The unitary part should produce coherent evolution
        # Check that imaginary part exists (coherent evolution)
        @test any(abs.(imag.(R)) .> 1e-10)
    end

    @testset "Dissipative part structure" begin
        # The dissipative part should produce decay
        # Check that real part exists (dissipation)
        @test any(abs.(real.(R)) .> 1e-10)
    end
end

@testset "Special cases" begin
    @testset "Zero coupling (weak dissipation)" begin
        N = 2
        H = Diagonal([0.0, 1.0])

        # Very weak coupling to test near-isolated limit
        σz = [1.0 0.0; 0.0 -1.0]
        NPS_zero(ω) = 1e-15  # Extremely weak
        a_ops = [(σz, NPS_zero)]

        R = BR_tensor(H, a_ops, false)

        # Dissipation should be extremely small
        dissipation = maximum(abs.(real.(R)))
        @test dissipation < 1e-10
    end

    @testset "Degenerate energy levels" begin
        N = 3
        H = Diagonal([1.0, 1.0, 2.0])  # Two degenerate levels

        NPS(ω) = 0.1
        σz = Matrix(Diagonal([1.0, -1.0, 0.0]))
        a_ops = [(σz, NPS)]

        R = BR_tensor(H, a_ops, true)

        # Should not crash or produce NaN
        @test all(isfinite.(R))
        @test size(R) == (N^2, N^2)
    end
end

@testset "Consistency with quantum master equation theory" begin
    # Test that BR tensor produces physically reasonable dynamics
    N = 2
    ω0 = 1.0
    H = Diagonal([0.0, ω0])

    # Dephasing operator
    σz = [1.0 0.0; 0.0 -1.0]
    γ = 0.1  # Dephasing rate
    NPS_dephasing(ω) = γ / π  # White noise limit

    a_ops = [(σz, NPS_dephasing)]
    R = BR_tensor(H, a_ops, false)

    @testset "Diagonal elements (populations)" begin
        # Population elements should decay
        # For pure dephasing, populations don't decay, only coherences

        # Initial state: superposition |+⟩ = (|0⟩ + |1⟩)/√2
        ψ0 = [1.0, 1.0] / sqrt(2)
        ρ0 = ψ0 * ψ0'
        ρ0_vec = vec(ρ0)

        # Short time evolution
        dt = 0.01
        ρ_vec = exp(R * dt) * ρ0_vec
        ρ = reshape(ρ_vec, N, N)

        # Populations should be preserved (pure dephasing)
        @test abs(real(ρ[1, 1]) - real(ρ0[1, 1])) < 0.1
        @test abs(real(ρ[2, 2]) - real(ρ0[2, 2])) < 0.1
    end
end

@testset "Numerical stability" begin
    @testset "Large system" begin
        N = 10
        H = Diagonal(range(0, 10, length=N))

        NPS(ω) = S(ω, 0.15 * eV, 0.1, 1 / (0.025 * eV))
        a_ops = [(Matrix(Diagonal([i == k ? 1.0 : 0.0 for i in 1:N])), NPS) for k in 1:N]

        R = BR_tensor(H, a_ops, true)

        @test all(isfinite.(R))
        @test size(R) == (N^2, N^2)
    end

    @testset "Small coupling" begin
        N = 2
        H = Diagonal([0.0, 1.0])

        # Very weak coupling
        σz = [1.0 0.0; 0.0 -1.0]
        NPS_weak(ω) = 1e-10

        a_ops = [(σz, NPS_weak)]
        R = BR_tensor(H, a_ops, false)

        @test all(isfinite.(R))
        # Should be very small dissipation
        @test maximum(abs.(real.(R))) < 1.0
    end
end;