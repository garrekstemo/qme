using LinearAlgebra
using GLMakie

# Partial trace of bipartite systems
function partial_trace(ρ, d1, d2, system=1)
    ρ_4d = reshape(ρ, (d1, d2, d1, d2))  # Reshape to 4D tensor

    # sum over the subsystem to be traced out
    if system == 1
        return sum(ρ_4d[i, :, i, :] for i in 1:d1)  # Keep system 2
    else
        return sum(ρ_4d[:, j, :, j] for j in 1:d2)  # Keep system 1
    end
end

d1, d2 = 2, 2  # dimension of each subsystem
θs = range(0, π/2, length = 100)  # angle for superposition coefficients
purity = []  # purity set

for θ in θs
    ψ = cos(θ) * kron(I(d1)[:, 1], I(d2)[:, 1]) + sin(θ) * kron(I(d1)[:, 2], I(d2)[:, 2])  # state vector
    ρ = ψ * ψ'  # density operator associated with ψ
    ρ1 = partial_trace(ρ, d1, d2, 2)  # marginal state of system 1
    push!(purity, tr(ρ1^2))  # calculate trace and append purity
end

set_theme!(theme_latexfonts())
fig = Figure(fontsize = 18)
ax = Axis(fig[1, 1],
    xlabel = "θ/π",
    ylabel = L"Purity $\mathcal{P}$[ρ_1(θ)]",
    xticks = LinearTicks(6),
    yticks = LinearTicks(6),
)
lines!(θs / π, purity)
fig