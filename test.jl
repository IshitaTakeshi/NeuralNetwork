using Formatting
import Base.error
using Base.Test

include("feedforward.jl")

function test_sigmoid()
    sigmoid = Sigmoid()
    @test sigmoid.activation([1, -1]) == [
        0.7310585786300049,
        0.2689414213699951
    ]
    @test sigmoid.dactivation([1, -1]) == [
        0.19661193324148185,
        0.19661193324148185
    ]
end


function linear_network()
    net = Network(2, 3, 2, Linear(), 0.04)

    net.W₁ = [
        1. 1.;
        0. 1.;
        1. 0.;
    ]
    net.b₁ = [
        1.,
        0.,
        0.
    ]

    net.W₂ = [
        1. 1. 0.;
        0. 1. 1.;
    ]
    net.b₂ = [
        0.,
        1.,
    ]

    return net
end


function test_forward()
    net = linear_network()
    @test [2., 0., 1.] == layer1(net, [1., 0.])
    @test [2., 2.] == layer2(net, [2., 0., 1.])
end


function test_Δlayer2()
    net = linear_network()
    h = [2., 0., 1]

    Δw₂, Δb₂ = Δlayer2(net, h, 1, 1)

    @test Δw₂ == h
    @test Δb₂ == 1
end


function test_Δlayer1()
    net = linear_network()
    x = [1., 0.]

    Δw₂, Δb₂ = Δlayer1(net, x, 1, 2)

    @test Δw₂ == 2x
    @test Δb₂ == 2
end


# test_sigmoid()
# test_forward()
# test_Δlayer2()
# test_Δlayer1()


X = Float64[
    0 0 1;
    0 1 0;
]


Y = Float64[
    0 0 1;
    0 1 0;
]


# net = Network(2, 2, 2, Linear(), 0.008)

net = linear_network()

for i in 1:100
    net = backward(net, X, Y)
    Yₚ = forward(net, X)
    println(error(Y, Yₚ))
end

println("W₁")
println(net.W₁)
println("b₁")
println(net.b₁)

println("W₂")
println(net.W₂)
println("b₂")
println(net.b₂)

for i in 1:size(X, 2)
    x = view(X, :, i)
    println(x, forward(net, x))
end
