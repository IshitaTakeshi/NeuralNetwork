import Base.+


type Loss
    value::Real
    count_added::Int
end


Loss() = Loss(0, 0)


+(loss::Loss, delta::Real) = Loss(loss.value + delta, loss.count_added + 1)


function mean(loss::Loss)
    if loss.count_added == 0
        return 0
    end
    return loss.value / loss.count_added
end


value(loss::Loss) = loss.value
