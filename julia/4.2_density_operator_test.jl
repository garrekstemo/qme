using LinearAlgebra

function is_state(ρ)
    vals = eigvals(ρ)  # eigenvalues of ρ
    non_unit = 1 - tr(ρ)  # deviation from unit trace
    non_hermitian = norm(ρ - ρ')  # deviation from Hermiticity
    non_positive = sum((abs.(vals) .- vals) / 2)  # deviation from positivity

    # return 1 if ρ is a valid density matrix, else return a value < 1
    return 1 - norm([non_unit, non_hermitian, non_positive])
end

# a state
ρ_1 = [
    0.2 0 0
    0 0.3 0
    0 0 0.5]

# a state with some error
ρ_2 = ρ_1 + [
    -1e-4 1e-6 0
    0 0 1e-2im
    1e-3im 0 1e-4]

println("Valid state? (ρ_1): ", is_state(ρ_1))
println("Valid state? (ρ_2): ", is_state(ρ_2))
