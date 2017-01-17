module NeuralNetwork


typealias AA AbstractArray


include("utils.jl")
include("mlp.jl")

export ConvolutionLayer, LogisticRegression, fit!, predict
export onehot, maxindices


type ConvolutionLayer
    filterw::Int
    filterh::Int
    stepw::Int
    steph::Int
    weights::Matrix
    bias::Matrix

    function ConvolutionLayer(filter_size::Int, step_size::Int,
                              weights = zeros(filter_size, filter_size))
        filterw, filterh = to_tuple_if_int(filter_size)
        stepw, steph = to_tuple_if_int(step_size)
        ConvolutionLayer(filterw, filterh, stepw, steph, weights)
    end

    function ConvolutionLayer(filterw::Int, filterh::Int,
                              stepw::Int, steph::Int,
                              weights = zeros(filterw, filterh))
        new(filterw, filterh, stepw, steph, weights)
    end
end


function forward(layer::ConvolutionLayer, image::Matrix)
    """
    Forward computation for an image
    """
    filterw, filterh = layer.filterw, layer.filterh
    startpointsx = 1:layer.stepw:(size(image, 1)-filterw+1)
    startpointsy = 1:layer.steph:(size(image, 2)-filterh+1)

    output = zeros(length(startpointsx), length(startpointsy))
    for (ox, x) in enumerate(startpointsx)
        for (oy, y) in enumerate(startpointsy)
            subimage = image[x:x+filterw-1, y:y+filterh-1]
            output[ox, oy] = sum(subimage .* layer.weights)
        end
    end
    output += layer.bias
    return output
end


function backward(layer::ConvolutionLayer)
end

end # module
