sigmoid(x) = 1 ./ (1 .+ e.^(-x))

function ∇sigmoid(x)
    f = sigmoid(x)
    return f .* (1 - f)
end
