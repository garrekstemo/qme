using LinearAlgebra

"""
    onehot(n, k)

Create a one-hot encoded vector of length `n` with a 1 at position `k`.
"""
function onehot(n, k)
    if n < k
        throw(ArgumentError("n must be >= k (got n=$n, k=$k)"))
    end
    v = zeros(Int, n)
    v[k] = 1
    v
end

"""
    offdiagonal_ones(n)

Create an n×n tridiagonal matrix with ones on the super- and sub-diagonals
and zeros on the main diagonal. This is equivalent to a Toeplitz matrix
constructed from [0, 1, 0, ...].

# Example
```julia
offdiagonal_ones(3)
# Returns:
# 0  1  0
# 1  0  1
# 0  1  0
```
"""
function offdiagonal_ones(n)
    Tridiagonal(ones(n-1), zeros(n), ones(n-1))  # from LinearAlgebra
end
