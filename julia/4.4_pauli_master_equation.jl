using LinearAlgebra
using GLMakie
using StatsBase
using Random
include("src/bloch_redfield_tensor.jl")

# constants and units
const fs = 1 / (2.4189e-2)  # femtosecond in Hartree AU
const eV = 1 / (fs * 0.6582)  # electronvolt in Hartree AU

Random.seed!(12)

NPS(ω) = S(ω, 0.15 * eV, 0.1, 1 / (0.025 * eV))

function simulate_dynamics(dts, A, ρ, X, num_steps=1000)
    t = 0.0
    times = Float64[]
    positions = Float64[]
    for dt in dts
        P = exp(A * dt)
        for m in 1:num_steps
            push!(times, t)
            push!(positions, real(tr(X' * ρ)))
            ρ = P * ρ
            t += dt
        end
    end
    return times, positions
end

function construct_W(A_operators, vals, kets, N)
    W = zeros(N, N)
    for (A_op, nps) in A_operators
        A = kets' * A_op * kets
        for a in 1:N
            W[a, a] -= sum([A[a, b] * A[b, a] * nps(vals[a] - vals[b]) for b in 1:N])
            for b in 1:N
                W[a, b] += A[b, a] * A[a, b] * nps(vals[b] - vals[a])
            end
        end
    end
    return W
end

N = 10  # number of sites

# disorder parameters
σ_E = 0.1 * eV
σ_V = 0.05 * eV

# Random energies
Es = σ_E * randn(N)

# Random couplings
all_pairs = [(j, k) for j in 1:N for k in j+1:N]
indices = sample(1:length(all_pairs), N, replace=false)
pairs = all_pairs[indices]
Vs = [(i, j, σ_V * randn()) for (i, j) in pairs]

# Construct Hamiltonian
H = Matrix(Diagonal(Es))
for (j, k, v) in Vs
    H[j, k] = v
    H[k, j] = conj(v)
end

vals, kets = eigen(H)
proj(k, N) = Matrix(Diagonal([i == k ? 1.0 : 0.0 for i in 1:N]))
a_ops = [(proj(k, N), NPS) for k in 1:N]

R = BR_tensor(H, a_ops)


# Calculate dynamics for Bloch-Redfield master equation
# and Pauli master equation

dts = [1e-1, 1e1, 1e3] * fs  # adaptive time scales

# simulate Bloch-Redfield dynamics
X = Diagonal(1:N)
X_vec = vec(kets' * X * kets)
ρ0 = zeros(N, N)
ρ0[1, 1] = 1.0  # initial state
ρ_vec = vec(kets' * ρ0 * kets)
times, positions = simulate_dynamics(dts, R, ρ_vec, X_vec)


# simulate Pauli master equation dynamics
W = construct_W(a_ops, vals, kets, N)
p0 = diag(kets' * ρ0 * kets) # initial state
x = diag(kets' * X * kets) # position operator
times_p, positions_p = simulate_dynamics(dts, W, p0, x, 1000)


fig = Figure()
DataInspector()
ax = Axis(fig[1, 1],
    title="Random Quantum Network Exciton Dynamics",
    xlabel="Time (fs)",
    ylabel="Position (a.u.)",
    xscale=log10,
)
lines!(times ./ fs, positions, label="Bloch-Redfield ME")
lines!(times_p ./ fs, positions_p, linestyle=:dash, label="Pauli ME")

xlims!(1e-1, 1e7)
axislegend(ax, position=:rb)

fig