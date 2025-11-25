using LinearAlgebra

"""
    liouvillian(H, Ls; ħ=1)

Constructs the Liouville superoperator (Liouvillian) from the Hamiltonian `H`
and a collection of Lindblad operators `Ls` (already scaled by √γ).

The Liouvillian describes the evolution of density matrices in the Lindblad
master equation: dρ/dt = ℒ[ρ], where ℒ is the Liouvillian superoperator.

# Mathematical form
ℒ[ρ] = -i/ħ [H, ρ] + ∑ᵢ (Lᵢ ρ Lᵢ† - 1/2{Lᵢ†Lᵢ, ρ})

In vectorized form (with ρ → vec(ρ)):
ℒ = -i/ħ (I ⊗ H - Hᵀ ⊗ I) + ∑ᵢ (L̄ᵢ ⊗ Lᵢ - 1/2(I ⊗ Lᵢ†Lᵢ + (Lᵢ†Lᵢ)ᵀ ⊗ I))

# Arguments
- `H`: Hamiltonian matrix (d×d)
- `Ls`: Collection of Lindblad operators (each d×d), pre-scaled by √γ
- `ħ`: Reduced Planck constant (default: 1)

# Returns
- Liouvillian superoperator matrix (d²×d²)
"""
function liouvillian(H, Ls; ħ=1)
    d = size(H, 1)

    # Hamiltonian part: -i/ħ [H, ρ] → -i/ħ (I⊗H - Hᵀ⊗I) vec(ρ)
    superH = -im/ħ * (kron(I(d), H) - kron(transpose(H), I(d)))

    # Lindblad dissipator part
    if isempty(Ls)
        # No dissipation
        superL = zeros(ComplexF64, d^2, d^2)
    else
        # ∑ᵢ (L̄ᵢ⊗Lᵢ - 1/2(I⊗Lᵢ†Lᵢ + (Lᵢ†Lᵢ)ᵀ⊗I))
        superL = sum(L -> kron(conj(L), L) - 0.5 * (kron(I(d), L'*L) + kron(transpose(L'*L), I(d))), Ls)
    end

    superH + superL
end
