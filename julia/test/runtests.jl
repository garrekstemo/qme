# These are just some basic checks and not a full test suite
# The Project.toml for the Julia code does not include
# a name or UUID so `runtests` from the REPL will not work.

using LinearAlgebra
using Test
using ToeplitzMatrices

@testset verbose = true "QME.jl Tests" begin
    include("test_helper_functions.jl")
    include("test_floquet.jl")
    include("test_floquet_decoherence.jl")
    include("test_random_quantum_network.jl")
    include("test_pauli_master_equation.jl")
    include("test_bloch_redfield_comparison.jl")
end;  # QME.jl Tests