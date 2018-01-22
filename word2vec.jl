include("utils.jl")


const AA = AbstractArray

srand(1234)


type Word2Vec
    n_context::Int
    n_embedding::Int
    η::Float64
    Wᵢ::Matrix{Float64}
    Wₒ::Matrix{Float64}
    n_vocabulary::Int
    function Word2Vec(;n_context = 5, n_embedding = 300, η = 0.01)
        new(n_context, n_embedding, η, zeros(0, 0), zeros(0, 0), 0)
    end
end


function embed(words)
    indices = findin(vocabulary, words)
    W = Matrix{Float64}(model.n_embedding, length(words))
    for (i, index) in indices
        if index == 0
            throw("Word not found in the vocabulary")
        end
        W[:, i] = Wᵢ[:, index]
    end
end


function save(model::Word2Vec)

end


function load()
    Word2Vec()
end


function init_weights(model::Word2Vec)
    model.Wᵢ = randn(model.n_embedding, model.n_vocabulary)
    model.Wₒ = randn(model.n_vocabulary, model.n_embedding)
    return model
end


function train!{T <: AbstractString}(model::Word2Vec, words::AA{T, 1};
                                     n_epochs=1)
    vocabulary = unique(words)
    indices = findin(vocabulary, words)  # word -> word index
    model.n_vocabulary = length(vocabulary)

    model = init_weights(model)

    for i in 1:n_epochs
        loss = epoch!(model, indices)
        println(loss)
    end
    return model
end


contextof(i, n, indices) = vcat(indices[i-n:i-1], indices[i+1:i+n])


function train_set_generator(n, indices)
    ((indices[i], contextof(i, n, indices)) for i in n+1:length(indices)-n)
end


function epoch!(model::Word2Vec, indices::AA{Int, 1})
    loss = Loss()
    for (word_index, context) in train_set_generator(model.n_context, indices)
        loss += update!(model, word_index, context)
    end
    return mean(loss)
end


function softmax(x)
    z = exp.(x)
    return z ./ sum(z)
end


function one_hot(word)
    vector = zeros(length(vocabulary))
    vector[index(word)] = 1
    return vector
end


cross_entropy_loss(word_index, y) = -log(y[word_index])


ΔWₒ(h::AA{Float64, 1}, Δ::AA{Float64, 1}) = Δ * h'


Δwᵢ(Wₒ::AA{Float64, 2}, Δ::AA{Float64, 1}) = Wₒ' * Δ


function update!(model::Word2Vec, word_index::Int, context::AA{Int, 1})
    loss = Loss()
    for context_index in context
        t = zeros(model.n_vocabulary)
        t[context_index] = 1

        h = view(model.Wᵢ, :, word_index)  # equivalent to h = model.Wᵢ * x
        y = softmax(model.Wₒ * h)

        Δ = y - t
        model.Wₒ -= model.η * ΔWₒ(h, Δ)
        model.Wᵢ[:, word_index] -= model.η * Δwᵢ(model.Wₒ, Δ)

        loss += cross_entropy_loss(word_index, y)
    end
    return mean(loss)
end
