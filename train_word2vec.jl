include("word2vec.jl")

text8 = readstring("text8")
text8 = split(strip(text8), " ")[1:2000]


model = Word2Vec(η = 0.009)
model = train!(model, text8, n_epochs=5)
