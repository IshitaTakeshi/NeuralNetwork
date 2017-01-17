function onehot(y::Vector)
    n_samples = size(y, 1)
    n_features = length(unique(y))
    Y = zeros(n_features, n_samples)

    # all elements of y must be > 0 because they are treated
    # as indices
    assert(all(0 .< y))
    for i in 1:n_samples
        Y[y[i], i] = 1
    end
    return Y
end


maxindices{T}(X::AA{T, 2}) = [indmax(X[:, i]) for i in 1:size(X, 2)]


function accuracy{T}(y::AA{T, 1}, ŷ::AA{T, 1})
    assert(size(y) == size(ŷ))
    sum(y .== ŷ) / length(y)
end


to_tuple_if_int(x::Real) = (x, x)
