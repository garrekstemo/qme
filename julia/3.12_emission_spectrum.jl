using LinearAlgebra
using FFTW
using GLMakie
include("src/liouvillian.jl")

Ω = 1 # Rabi frequency
Γ = 0.1 # decay rate
σm = [0 0; 1 0]  # emission operator
σp = σm'  # absorption operator
H = [0 Ω; Ω 0] / 2  # system Hamiltonian
collapse = [sqrt(Γ) * σm]  # collapse operators

L = liouvillian(H, collapse)  # Liouvillian superoperator
ρ_ss = reshape(nullspace(L), 2, 2)  # steady state density matrix
ρ_ss /= tr(ρ_ss)  # normalized steady state
d = size(ρ_ss, 1)  # system dimension

# correlation function
N = 2000  # samples
times = range(0, 500, length = N)  # time interval
corrs = zeros(ComplexF64, N)  # correlation array
dt = times[2] - times[1]  # time-step finite difference
P = exp(L * dt)  # propagator for one time step
B_ss = σm * ρ_ss  # emission operator applied to steady state
B_ss_vec = vec(B_ss)  # vectorize B_ss

# calculate correlation function over the time interval
for (kt, t) in enumerate(times)
    corrs[kt] = tr(σp * reshape(B_ss_vec, d, d))  # collect correlation
    B_ss_vec = P * B_ss_vec  # propagate operator using semigroup composition rule
end

# Obtain correlation spectrum using discrete Fourier transform
spectrum = 2 * real(fft(corrs)) * dt
Ω_list = 2π * fftfreq(N, 1/dt)  # angular frequencies - note: fftfreq expects sampling rate, not spacing!


fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1],
    title = "Correlation Function",
    xlabel = "t / Ω",
    ylabel = "C(t)",
    xticks = LinearTicks(6),
    yticks = 0:0.1:0.5,
    )
lines!(times, real(corrs))
lines!(times, imag(corrs), linestyle = :dash)

ax2 = Axis(fig[2, 1], 
    title = "Emission Spectrum",
    xlabel = "ω / Ω",
    ylabel = "E(ω)",
    )
scatter!(Ω_list, spectrum, label = "Semigroup")

ggee = "|g⟩ → |g⟩\n|e⟩ → |e⟩"
ge = "|g⟩ → |e⟩"
eg = "|e⟩ → |g⟩"
text!(ggee, position = (0.5, 0.85), align = (:center, :center), space = :relative)
text!(ge, position = (0.25, 0.3), align = (:center, :center), space = :relative)
text!(eg, position = (0.75, 0.3), align = (:center, :center), space = :relative)

xlims!(ax1, 0, 100)
xlims!(ax2, -2, 2)
ylims!(ax2, -0.2, 18)
fig