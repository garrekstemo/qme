"""
Unit tests for Floquet decoherence implementation.

Tests the Julia floquet_decoherence() function against the Python
reference implementation and checks physical properties.
"""

using Test
using LinearAlgebra
using Test
using LinearAlgebra
using PythonCall

# Import Julia implementation
include("../src/floquet_decoherence.jl")
include("../src/liouvillian.jl") # Needed for liouvillian() helper

# Import Python implementation
sys = pyimport("sys")
python_path = abspath(joinpath(@__DIR__, "../../python"))
if !(python_path in sys.path)
    sys.path.append(python_path)
end
py_floquet = pyimport("floquet_decoherence")
py_superop = pyimport("superoperator")

@testset "Floquet Decoherence Tests" begin
    # Test parameters
    ϵ = 0.2
    Δ = -6.0
    ω = 1.5
    n_ph = 13
    Γz = 0.1
    times = [10.0 / Γz]

    # Pauli matrices
    σz = [1 0; 0 -1]
    σx = [0 1; 1 0]

    # Lindblad operators
    L = σz
    Ls_jl = [sqrt(Γz) * L]
    Ls_py = [sqrt(Γz) * L]

    # Hamiltonians
    H_0 = Δ / 2 * σz + ϵ * σx
    H_int = σz / 2

    # Liouvillians
    P0_jl = liouvillian(H_0, Ls_jl)
    P_int_jl = liouvillian(H_int, [])

    # Import numpy
    np = pyimport("numpy")

    # Convert inputs to numpy arrays for Python
    H_0_py = np.array(H_0)
    H_int_py = np.array(H_int)
    Ls_py_numpy = [np.array(L) for L in Ls_py]

    P0_py_obj = py_superop.Liouvillian(H_0_py, Ls_py_numpy)
    P_int_py_obj = py_superop.Liouvillian(H_int_py, [])

    P0_py = pyconvert(Matrix{ComplexF64}, P0_py_obj)
    P_int_py = pyconvert(Matrix{ComplexF64}, P_int_py_obj)

    @testset "Liouvillian construction" begin
        @test P0_jl ≈ P0_py
        @test P_int_jl ≈ P_int_py
    end

    @testset "Floquet matrix dimensions" begin
        d = size(P0_jl, 1)
        n_max = floor(Int, n_ph / 2)

        # Photon Hamiltonian
        H_ph_jl = ω * kron(Diagonal(-n_max:n_max), I(d))

        # Full Floquet matrix
        Pf_jl = kron(I(n_ph), im * P0_jl) + H_ph_jl + kron(offdiagonal_ones(n_ph), im * P_int_jl)

        @test size(Pf_jl) == (n_ph * d, n_ph * d)
        @test size(Pf_jl) == (52, 52)
    end

    @testset "Propagator comparison (Python vs Julia)" begin
        # Non-averaged
        U_t_jl = floquet_decoherence(times, ω, n_ph, P0_jl, P_int_jl, false)
        U_t_py_obj = py_floquet.floquet_decoherence(times, ω, n_ph, P0_py_obj, P_int_py_obj, false)
        U_t_py = pyconvert(Array{ComplexF64,3}, U_t_py_obj)

        @test size(U_t_jl) == size(U_t_py)
        @test U_t_jl[:, :, 1] ≈ U_t_py[:, :, 1] rtol = 1e-10

        # Averaged
        U_t_jl_avg = floquet_decoherence(times, ω, n_ph, P0_jl, P_int_jl, true)
        U_t_py_avg_obj = py_floquet.floquet_decoherence(times, ω, n_ph, P0_py_obj, P_int_py_obj, true)
        U_t_py_avg = pyconvert(Array{ComplexF64,3}, U_t_py_avg_obj)

        @test size(U_t_jl_avg) == size(U_t_py_avg)
        @test U_t_jl_avg[:, :, 1] ≈ U_t_py_avg[:, :, 1] rtol = 1e-10
    end

    @testset "Physical Properties" begin
        # Initial state (Ground state)
        vals, vecs = eigen(H_0)
        ρ0 = vecs[:, 1] * vecs[:, 1]'
        ρ0_vec = reshape(ρ0, :, 1)

        # Evolve
        U_t = floquet_decoherence(times, ω, n_ph, P0_jl, P_int_jl, false)
        ρ_t_vec = U_t[:, :, 1] * ρ0_vec
        ρ_t = reshape(ρ_t_vec, 2, 2)

        # 1. Trace preservation
        @test tr(ρ_t) ≈ 1.0 atol = 1e-10

        # 2. Hermiticity
        @test ρ_t ≈ ρ_t' atol = 1e-10

        # 3. Positivity (eigenvalues >= 0)
        evals = eigvals(ρ_t)
        @test all(real(evals) .>= -1e-10)
        @test sum(imag(evals)) ≈ 0.0 atol = 1e-10
    end

    @testset "Multiple time points" begin
        times_multi = [10.0 / Γz, 20.0 / Γz, 30.0 / Γz]
        U_t_jl = floquet_decoherence(times_multi, ω, n_ph, P0_jl, P_int_jl, false)

        @test size(U_t_jl) == (4, 4, 3)

        # Check trace preservation for all times
        vals, vecs = eigen(H_0)
        ρ0 = vecs[:, 1] * vecs[:, 1]'
        ρ0_vec = reshape(ρ0, :, 1)

        for i in 1:3
            ρ_t = reshape(U_t_jl[:, :, i] * ρ0_vec, 2, 2)
            @test tr(ρ_t) ≈ 1.0 atol = 1e-10
        end
    end

    @testset "Error handling" begin
        @test_throws ErrorException floquet_decoherence(times, ω, 12, P0_jl, P_int_jl, false)
    end
end
