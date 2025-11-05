using LinearAlgebra

# LinearAlgebra is only needed for identity matrix `I`.
# Note that `I(n)` creates an n x n identity matrix,
# but `I` acts like an identity matrix for any compatible size.

d_a, d_b = 2, 3 # dimensions of the bases
basis = [I zeros(d_a, d_b); zeros(d_b, d_a) I] # block matrix