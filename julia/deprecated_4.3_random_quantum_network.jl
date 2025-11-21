using LinearAlgebra
using GLMakie
using PythonCall

include("src/bloch_redfield_tensor.jl")

# constants and units
fs = 1 / (2.4189e-2)  # femtosecond in Hartree AU
eV = 1 / (fs * 0.6582)  # electronvolt in Hartree AU

N = 10  # number of sites

# disorder parameters
σ_E = 100e-3 * eV
σ_V = 50e-3 * eV

# ============================================================================
# Random number generation: Two equivalent implementations
# ============================================================================
# For exact reproducibility with Python code, use Python's RNG (Option 1).
# For standalone Julia code, use Julia's native RNG (Option 2).
# Both produce equivalent random network structures.
# ============================================================================

# --- Option 1: Python RNG (for exact Python/Julia matching) ---
const np = pyimport("numpy")
np.random.seed(0)

# Random energies (using Python's RNG)
Es = pyconvert(Vector{Float64}, np.random.normal(0, σ_E, N))

H = Matrix(Diagonal(Es))  # Hamiltonian

# Random couplings (using Python's RNG)
# Pattern: [(j, k) for j in 1:N for k in j+1:N] matches Python's
# [(x, y) for x in range(N) for y in range(x+1, N)]
all_pairs = [(j, k) for j in 1:N for k in j+1:N]
indices_py = np.random.choice(pylist(0:length(all_pairs)-1), N, replace=false)
indices = [pyconvert(Int, i) + 1 for i in indices_py]  # Convert to 1-based indexing

for idx in indices
    j, k = all_pairs[idx]
    v = pyconvert(Float64, np.random.normal(0, σ_V))
    H[j, k] = v
    H[k, j] = conj(v)  # Hermiticity
end

# --- Option 2: Pure Julia RNG (standalone, no Python dependency) ---
# Uncomment to use Julia's native random number generator:
#=
using Random
Random.seed!(0)

# Random energies (using Julia's RNG)
Es = σ_E * randn(N)

H = Matrix(Diagonal(Es))  # Hamiltonian

# Random couplings (using Julia's RNG)
# Same pattern as Python version, just with Julia's RNG
all_pairs = [(j, k) for j in 1:N for k in j+1:N]
using StatsBase
indices = sample(1:length(all_pairs), N; replace=false)

for idx in indices
    j, k = all_pairs[idx]
    v = σ_V * randn()
    H[j, k] = v
    H[k, j] = conj(v)  # Hermiticity
end
=#

# ============================================================================

vals, kets = eigen(Hermitian(H))

NPS(ω) = S(ω, 150e-3 * eV, 1e-1, 1/(25e-3 * eV))

# site projection operators
proj(k, N) = Matrix(Diagonal([i == k ? 1.0 : 0.0 for i in 0:N-1]))
a_ops = [(proj(k, N), NPS) for k in 0:N-1]

R = BR_tensor(H, a_ops)

# position operator
X = Diagonal(0:N-1)  # Match Python's 0-based indexing
X_vec = reshape(kets' * X * kets, N^2)
ρ = zeros(N, N)
ρ[1, 1] = 1.0 # initial state
ρ_vec = reshape(kets' * ρ * kets, N^2)

dts = [1e-1, 1e1, 1e3] * fs

function simulate_dynamics(dts, ρ_vec, X_vec)
    t = 0
    times = []
    positions = []
    for dt in dts
        P = exp(R * dt)
        for m in 1:1000
            if t > 0
                push!(times, t)
                push!(positions, real(ρ_vec' * X_vec))
            end
            ρ_vec = P * ρ_vec
            t += dt
        end
    end
    return times, positions
end