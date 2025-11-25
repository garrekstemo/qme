include("liouvillian.jl")
include("helper_functions.jl")

"""
    floquet_decoherence(times, ω, n_photons, P0, P_int, average=false)

Obtain the Floquet propagator U(t) in terms of Floquet blocks for a harmonically driven system.
U -> U(t) = ( ..., U[2], U[1], U[0], U[-1], U[-2], ...)

# Arguments
- `times`: Vector of time steps.
- `ω`: Driving frequency.
- `n_photons`: Number of photons driving the interaction (must be odd).
- `P0`: Time-independent part of the Liouville superoperator.
- `P_int`: Superoperator of the interaction.
- `average`: If `true`, average over all possible phases of the driving field.
"""
function floquet_decoherence(times, ω, n_photons, P0, P_int, average=false)
    if n_photons % 2 == 0
        error("n_photons must be odd")
    end

    P0 = im * P0
    P_int = im * P_int

    d = size(P0, 1)
    U0 = I(d)  # initial generator

    n_max = floor(Int, n_photons / 2)
    H_ph = ω * kron(Diagonal(-n_max:n_max), I(d))

    # system
    Pf = kron(I(n_photons), P0) + H_ph + kron(offdiagonal_ones(n_photons), P_int)

    vals, vecs = eigen(Pf)

    U0_f = kron(onehot(n_photons, n_max + 1)', U0)
    U_t = zeros(ComplexF64, d, d, length(times))

    for i in eachindex(times)
        U_eigs = Diagonal(exp.(-im * vals * times[i]))
        U_f = vecs * U_eigs * inv(vecs) * U0_f'

        if average
            U_t[:, :, i] += U_f[(1:d).+n_max*d, :]
        else
            for j in -n_max:n_max
                counter = j + n_max
                U_t[:, :, i] += U_f[(1:d).+counter*d, :] * exp(im * j * ω * times[i])
            end
        end
    end
    return U_t
end