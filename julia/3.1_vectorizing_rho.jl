using LinearAlgebra

# Note that Julia uses column-major order,
# so the result will differ from that in Python (which uses row-major order).
# Any reshaping is valid as long as it is consistent with subsequent operations.

ψ = [1, 2im, 0, -2, 0] # some (non-normalized) state
ρ = ψ * ψ'  # its density matrix
ρ /= tr(ρ)  # normalized density matrix
d = length(ψ)  # the dimension of the system's space
ρ_vec = vec(ρ) # vectorized density matrix