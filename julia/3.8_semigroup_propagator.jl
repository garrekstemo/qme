using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

H = [0 1; 1 0.5]
c_ops = [[0 1; 0 0]]

superop = liouvillian(H, c_ops)
ρ0 = [1 0; 0 0]

dt = 0.5  # time step
time_steps = 20
d = size(ρ0, 1)  # dimension of state
P = exp(superop * dt)  # propagator
times_0, populations_0 = [], []  # time and population arrays
t = 0.0
ρ = ρ0

# propagate
for k in 1:time_steps
    push!(populations_0, real(tr(ρ * ρ0)))  # append the current population
    push!(times_0, t)  # append the current time
    ρ = reshape(P * vec(ρ), d, d)  # propagate the state
    t += dt
end

function propagate(ρ0, superop, t)
    ρt_vec = exp(superop * t) * vec(ρ0)  # apply propagator to vectorized initial state
    return reshape(ρt_vec, d, d)  # return ρ(t) as a matrix
end

# time steps
times_1 = range(0, 10, 100)

# Population dynamics of the initial state
populations_1 = [real(tr(propagate(ρ0, superop, t) * ρ0)) for t in times_1]

# Plot
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "t", ylabel = "Tr[ρ(t)ρ(0)]")

lines!(times_1, populations_1, label = "w/ exp")
scatter!(times_0, populations_0,
    alpha=0.0,
    strokecolor = :black,
    strokewidth = 1.5,
    markersize = 12,
    label = "w/ semigroup"
)

axislegend(ax)
fig