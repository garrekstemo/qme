using LinearAlgebra
using DifferentialEquations
using GLMakie
include("src/liouvillian.jl")

# Parameters
Ω = 0.05
Γ = Ω / 5

# Build the superoperator
H_s = [0 Ω
       Ω 0]
d = size(H_s, 1)
L = sqrt(Γ) * [0 1; 1 0]
superop = liouvillian(H_s, [L])

# Here we must explicitly declare the state as complex
# to work with the DifferentialEquations.jl package
ρ0_vec = ComplexF64[0, 0, 0, 1]

# First repeat the SVD approach from 3.6
ts = range(0, 200, length = 200)
ρt_vec = zeros(length(ρ0_vec), length(ts))

for k in 1:d^2
    norm = left[:, k]' * right[:, k]
    ρt_vec += (left[:, k]' * ρ0_vec * right[:, k]) * exp.(λ[k] * ts)' / norm
end

# Now do the finite difference approach using DifferentialEquations.jl
t0 = 0.0
tf = 200.0
times = range(t0, tf, 20)

# Runga-Kutta method implemented in SciML's DifferentialEquations.jl package
f(u, p, t) = superop * u
tspan = (t0, tf)

prob = ODEProblem(f, ρ0_vec, tspan)
sol = solve(prob, RK4(), saveat = times)


fig = Figure()
ax = Axis(fig[1, 1], xlabel = "time (tΩ)", ylabel = "Tr[ρ(t)ρ₀]")
lines!(ts * Ω, real(ρt_vec[4, :]), label = "SVD")
scatter!(times .* Ω, real.(sol[4, :]), marker = :rect, color = :black, label = "RK4")

axislegend(ax)
fig