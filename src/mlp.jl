include("activation.jl")


type LogisticRegression
    n_input::Int
    n_output::Int
    η::Float64  # learning rate
    maxiter::Int
    error_threshold::Float64
    W::AA
    b::AA  # mean along axis 2
    X::AA
    Ŷ::AA  # Output of LogisticRegression Layer

    function LogisticRegression(n_input::Int,
                                   n_output::Int,
                                   learning_rate::Float64,
                                   maxiter::Int,
                                   error_threshold::Float64 = 1e-3,
                                   W::AA = zeros(n_output, n_input),
                                   b::AA = zeros(n_output))
        X = zeros(0, 0)
        Ŷ = zeros(0, 0)
        new(n_input, n_output, learning_rate, maxiter, error_threshold,
            W, b, X, Ŷ)
    end
end


forward{T}(W::AA{T, 2}, x::AA{T, 1}, b::AA{T, 1}) = sigmoid(W*x + b)


∇bias(y::Float64, ŷ::Float64) = (ŷ - y) * ŷ * (1 - ŷ)


∇weights(x::AA, y::Float64, ŷ::Float64) = (ŷ - y) * ŷ * (1 - ŷ) * x


function ∇bias(y::AA, ŷ::AA)
    assert(size(y) == size(ŷ))
    ∇b = zeros(size(y))
    for i in 1:size(y, 1)
        ∇b[i] = ∇bias(y[i], ŷ[i])
    end
    return ∇b
end


function ∇weights(x::AA, y::AA, ŷ::AA)
    assert(size(y) == size(ŷ))
    ∇W = zeros(size(y, 1), size(x, 1))
    for i in 1:size(y, 1)
        ∇W[i, :] = ∇weights(x, y[i], ŷ[i])
    end
    return ∇W
end


function forward{T}(W::AA{T, 2}, X::AA{T, 2}, b::AA{T, 1})
    N = size(X, 2)
    Ŷ = zeros(size(W, 1), N)
    for i in 1:N
        Ŷ[:, i] = forward(W, view(X, :, i), b)
    end
    return Ŷ
end


function forward!(layer::LogisticRegression, X::AA)
    layer.X = X
    layer.Ŷ = forward(layer.W, X, layer.b)
end


function backward!(layer::LogisticRegression, Y::AA)
    """
    Parameters
    ----------
    Y : AbstractArray
    Expected output represented as one-hot vectors
    """

    for i in 1:size(Y, 2)
        x = view(layer.X, :, i)
        y = view(Y, :, i)
        ŷ = view(layer.Ŷ, :, i)

        layer.W -= layer.η * ∇weights(x, y, ŷ)
        layer.b -= layer.η * ∇bias(y, ŷ)
    end
end


function fit!(layer::LogisticRegression, X::AA, Y::AA)
    assert(size(X, 2) == size(Y, 2))  # TODO rewrite with error()

    # TODO show progress
    for i in 1:layer.maxiter
        Ŷ = forward!(layer, X)
        backward!(layer, Y)
        if sum((Y - Ŷ) .^ 2) < layer.error_threshold
            return layer
        end
    end
    return layer
end


predict(layer::LogisticRegression, X::AA) = forward(layer.W, X, layer.b)
