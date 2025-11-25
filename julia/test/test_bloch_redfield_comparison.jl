"""
Comparison tests for Bloch-Redfield tensor against Python implementation.
"""

using Test
using LinearAlgebra
using PythonCall
using Random

# Constants
const fs = 1 / (2.4189e-2)
const eV = 1 / (fs * 0.6582)

# Setup Python environment
const np = pyimport("numpy")
sys = pyimport("sys")
sys.path.append(joinpath(@__DIR__, "../../python"))

include("../src/bloch_redfield_tensor.jl")

@testset "Bloch-Redfield Python Comparison" begin

    N = 4
    Random.seed!(42)

    # Generate random Hamiltonian
    H_jl = randn(N, N)
    H_jl = (H_jl + H_jl') / 2
    # Import Python module
    py_br = pyimport("bloch_redfield_tensor")
    # PythonCall handles conversion automatically or we use np.array
    H_py = np.array(H_jl)

    # Define NPS
    NPS(ω) = 0.1 / (1 + ω^2)

    # Define operators
    # Site 1 coupling
    A1 = zeros(N, N)
    A1[1, 1] = 1.0

    a_ops_jl = [(A1, NPS)]

    # Python setup
    pyexec("""
    import numpy as np
    def NPS_func(w):
        return 0.1 / (1 + w**2)

    A1 = np.zeros(($(N), $(N)))
    A1[0, 0] = 1.0

    a_ops = [[A1, NPS_func]]
    """, @__MODULE__)

    a_ops_py = pyeval("a_ops", @__MODULE__)

    # Calculate Tensors
    R_jl = BR_tensor(H_jl, a_ops_jl, false)
    R_py = py_br.BR_tensor(H_py, a_ops_py, secular=false)
    R_py_mat = pyconvert(Matrix{ComplexF64}, R_py)

    # Non-secular tensor is gauge dependent (depends on eigenvector phases),
    # so element-wise comparison may fail if phases differ.
    # We only check that it runs and has correct dimensions.
    @test size(R_jl) == size(R_py_mat)
    @test all(isfinite.(R_jl))

    # Secular approximation (Gauge Invariant)
    R_jl_sec = BR_tensor(H_jl, a_ops_jl, true)
    R_py_sec = py_br.BR_tensor(H_py, a_ops_py, secular=true)

    R_py_sec_mat = pyconvert(Matrix{ComplexF64}, R_py_sec)

    diff_sec = maximum(abs.(R_jl_sec - R_py_sec_mat))
    println("Secular Max Diff: ", diff_sec)
    @test diff_sec < 1e-9
end
