using LinearAlgebra
using GLMakie
include("src/liouvillian.jl")

σx = [0 1; 1 0]
σy = [0 -im; im 0]
σplus = (σx + im * σy) / 2
σminus = (σx - im * σy) / 2

superop_1 = liouvillian(σx, [σplus])
superop_2 = liouvillian(σy, [σminus])
ρ0 = [1 0; 0 0]
d = size(ρ0, 1) # dimension of the state

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "t", ylabel = "Tr[ρ(t)ρ(0)]")

ms = [50, 100, 1000]
alphas = [0.4, 0.7, 1.0]

for (km, m) in enumerate(ms)
    dt = 10 / m  # time step
    P_1 = exp(superop_1 * dt)
    P_2 = exp(superop_2 * dt)
    P = P_1 * P_2 # Suzuki-Trotter expansion for small times dt

    t = 0.0
    ρ = ρ0
    times_0 = []
    populations_0 = []
    for k in 1:m
        push!(populations_0, tr(ρ * ρ0))
        push!(times_0, t)
        ρ = reshape(P * vec(ρ), d, d)
        t += dt
    end

    lines!(ax, times_0, real(populations_0), color = :deepskyblue4, alpha = alphas[km], label = "m = $m")
end

axislegend(ax)
fig