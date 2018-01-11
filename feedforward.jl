using Formatting


const AA = AbstractArray

abstract type Activation end


type Sigmoid <: Activation
    activation::Function
    dactivation::Function

    function Sigmoid()
        activation(x) = 1 / (1 + exp(-x))
        dactivation(o) = o * (1 - o)
        new(activation, dactivation)
    end
end


type Linear <: Activation
    activation::Function
    dactivation::Function

    function Linear()
        activation(x) = x
        dactivation(o) = 1
        new(activation, dactivation)
    end
end


type Network
    n_input::Int
    n_hidden::Int
    n_output::Int

    W₁::AA{Float64, 2}
    b₁::AA{Float64, 1}

    W₂::AA{Float64, 2}
    b₂::AA{Float64, 1}

    activation::Function
    dactivation::Function
    η::Float64

    function Network(n_input::Int, n_hidden::Int, n_output::Int,
                     activation::Activation, η::Float64)
        W₁ = ones(Float64, n_hidden, n_input)
        b₁ = ones(Float64, n_hidden)

        W₂ = ones(Float64, n_output, n_hidden)
        b₂ = ones(Float64, n_output)

        new(n_input, n_hidden, n_output,
            W₁, b₁, W₂, b₂,
            activation.activation,
            activation.dactivation,
            η)
    end
end


function layer1(net::Network, x::AA{Float64, 1})
    net.activation.(net.W₁ * x + net.b₁)
end


function layer2(net::Network, h::AA{Float64, 1})
    net.activation.(net.W₂ * h + net.b₂)
end


function Δlayer1(net::Network, Δ::AA{Float64, 1}, o::AA{Float64, 1})
    (net.W₂' * Δ) .* net.dactivation.(o)
end


function Δlayer2(net::Network, Δ::AA{Float64, 1}, o::AA{Float64, 1})
    Δ .* net.dactivation.(o)
end


function forward(net::Network, x::AA{Float64, 1})
    layer2(net, layer1(net, x))
end


function backward(net::Network, x::AA{Float64, 1}, y::AA{Float64, 1})
    h = layer1(net, x)
    o = layer2(net, h)

    Δₒ = o - y
    Δ₂ = Δlayer2(net, Δₒ, o)
    Δ₁ = Δlayer1(net, Δ₂, h)
    printfmtln("Δ₂ : {: 1.3f} {: 1.3f}  h : {: 1.3f} {: 1.3f}", Δ₂..., h...)
    printfmtln("Δ₁ : {: 1.3f} {: 1.3f}  x : {: 1.3f} {: 1.3f}", Δ₁..., x...)

    net.W₂ -= net.η * Δ₂ * h'
    net.b₂ -= net.η * Δ₂

    net.W₁ -= net.η * Δ₁ * x'
    net.b₁ -= net.η * Δ₁
end


function forward(net::Network, X::AA{Float64, 2})
    N = size(X, 2)
    Y = Array{Float64, 2}(net.n_output, N)
    for i in 1:N
        Y[:, i] = forward(net, view(X, :, 1))
    end
    return Y
end


function update(net::Network, x::AA{Float64, 1}, y::AA{Float64, 1})
    for i in 1:net.n_output
        h = layer1(net, x)

        Δw₂, Δb₂ = Δlayer2(net, h, y[i], i)

        net.W₂[i, :] -= net.η * Δw₂
        net.b₂[i] -= net.η * Δb₂

        for j in 1:net.n_hidden
            Δw₁, Δb₁ = Δlayer1(net, x, y[i], j)
            net.W₁[j, :] -= net.η * Δw₁
            net.b₁[j] -= net.η * Δb₁
        end
    end

    return net
end


function backward(net::Network, X::AA{Float64, 2}, Y::AA{Float64, 2})
    for i in 1:size(Y, 2)
        backward(net, view(X, :, i), view(Y, :, i))
    end
    return net
end

error{T, U <: Real}(yₜ::AA{T, 1}, yₚ::AA{U, 1}) = dot(yₜ - yₚ, yₜ - yₚ) / 2


function error{T, U <: Real}(Yₜ::AA{T, 2}, Yₚ::AA{U, 2})
    sum(error(view(Yₜ, :, i), view(Yₚ, :, i)) for i in 1:size(Yₜ, 2))
end


function tocategorical(y::AA{Int, 1}, n_classes::Int)
    N = size(y, 1)
    Y = -ones(n_classes, N)

    for i in 1:N
        Y[y[i]+1, i]
    end

    return Y
end



