m = Float64[
    1 3 8 4;
    2 5 3 1;
    6 3 2 5;
]

f = Float64[
    2 1;
    1 3
]

expected = Float64[
    22 28 26;
    24 22 24
]

@test forward(ConvolutionLayer(2, 1, f), m) == expected


m = Float64[
    1 3 8;
    2 5 3;
    6 3 2;
]

f = Float64[
    2 1;
    1 3;
    1 0;
]
expected = Float64[
    28 31
]

output = forward(ConvolutionLayer(size(f, 1), size(f, 2), 1, 1, f), m)
@test output == expected
