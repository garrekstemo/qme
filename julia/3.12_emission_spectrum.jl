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
ax1 = Axis(fig[1, 1], xlabel = "t", ylabel = "Correlation")
lines!(ax1, times, real(corrs))
xlims!(ax1, 0, 100)

ax2 = Axis(fig[2, 1], xlabel = "ω", ylabel = "Spectrum")
lines!(ax2, Ω_list, spectrum)
xlims!(ax2, -2, 2)
fig