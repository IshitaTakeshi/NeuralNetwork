using Base.Test


include("utils.jl")


function test_add()
    loss = Loss()
    loss = loss + 1.0
    loss += 3
    @test loss.value == value(loss) == 4
    @test loss.count_added == 2
end


function test_mean()
    loss = Loss()
    loss += 3
    loss += 5
    # 2 times of increment and total value is 8
    @test mean(loss) == 4
end


test_add()
test_mean()
