"""
Unit tests for helper functions.
"""

using Test
using LinearAlgebra
using ToeplitzMatrices

include("../src/helper_functions.jl")

@testset "Helper Functions Tests" begin
    @testset "offdiagonal_ones" begin
        # Test basic functionality
        n = 3
        M = offdiagonal_ones(n)
        @test M == [
            0 1 0
            1 0 1
            0 1 0]

        # Test equivalence with Toeplitz
        a = [0, 1, 0]
        T = Toeplitz(a, a)
        @test M == T

        # Test larger matrix
        M5 = offdiagonal_ones(5)
        @test size(M5) == (5, 5)
        @test M5[1, 2] == 1
        @test M5[2, 1] == 1
        @test M5[1, 1] == 0
        @test M5[3, 3] == 0
    end

    @testset "onehot" begin
        n = 5
        for k in 1:n
            v = onehot(n, k)
            @test v == [i == k for i in 1:n]
        end
    end
end
