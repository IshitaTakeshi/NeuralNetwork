using Base.Test

include("word2vec.jl")


function test_contextof()
    indices = 1:12
    @test contextof(3, 2, indices) == [1, 2, 4, 5]
    @test contextof(5, 3, indices) == [2, 3, 4, 6, 7, 8]
end


function test_train_set_generator()
    indices = 1:7
    n = 2

    @test collect(train_set_generator(n, indices)) == Tuple{Int, Vector{Int}}[
       (3, [1, 2, 4, 5]),
       (4, [2, 3, 5, 6]),
       (5, [3, 4, 6, 7])
    ]
end


function test_ΔWₒ()
    h = [0.1, 0.4]
    y = [0.5, 0.1]
    t = [1.0, 0.0]
    @test ΔWₒ(h, y, t) ≈ [
        -0.05 -0.2;
        0.01 0.04;
    ]
end


function test_Δwᵢ()
    Wₒ = [
        0. 1. 1.;
        1. 1. 0.;
    ]
    y = [0.5, 0.1]
    t = [1.0, 0.0]

    @test Δwᵢ(Wₒ, y, t) ≈ [0.1, -0.4, -0.5]
end



test_contextof()
test_train_set_generator()
test_ΔWₒ()
test_Δwᵢ()
