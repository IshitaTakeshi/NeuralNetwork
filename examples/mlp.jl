using NeuralNetwork
import NeuralNetwork: accuracy

using RDatasets: dataset

iris = dataset("datasets", "iris")

X = Array(iris[:, 1:4])
y = zeros(Int, size(iris, 1))
for (i, label) in enumerate(["versicolor", "setosa", "virginica"])
    y[iris[:, 5] .== label] = i
end

indices = shuffle(1:size(iris, 1))

X = X[indices, :]'
y = y[indices]

N = Int(round(size(X, 2)*0.8))
X_train, X_test = X[:, 1:N], X[:, N+1:end]
y_train, y_test = y[1:N], y[N+1:end]

layer = LogisticRegression(4, 3, 5e-3, 1000)
fit!(layer, X_train, onehot(y_train))
Ŷ = predict(layer, X_test)

ŷ = maxindices(Ŷ)
println("Accuracy: $(accuracy(y_test, ŷ))")
