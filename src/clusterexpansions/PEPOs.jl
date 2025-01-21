function init_PEPO(pspace::ElementarySpace, trivspace::ElementarySpace)
    return Dict((0,0,0,0) => TensorMap([1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace'))   
end

function init_PEPO()
    return init_PEPO(ℂ^2, ℂ^1)
end

function init_PEPO(β, pspace::ElementarySpace, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    A = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
    A[][:,:,1,1,1,1] = exp(-β*onesite_op)[]
    return Dict((0,0,0,0) => A)
end

function init_PEPO(β, onesite_op::AbstractTensorMap)
    return init_PEPO(β, ℂ^2, ℂ^1, onesite_op)
end

function get_size_level(highest, highest_loop, spaces)
    return sum([dim(spaces(i)) for i = 0:highest]) + sum([dim(spaces(i)) for i = -1:-1:highest_loop])
end

function get_sum_space(highest, highest_loop, spaces; zerospace = ℂ^0)
    total_space = zerospace
    for ind = highest_loop:highest
        total_space = total_space ⊕ spaces(ind)
    end
    return total_space
end

function get_location_PEPO(ind, highest, spaces)
    if ind == 0
        return 1
    elseif ind < 0
        start = sum([dim(spaces(i)) for i = 0:highest]) + sum([dim(spaces(i)) for i = -1:-1:ind+1])
        return start+1:start+dim(spaces(ind))
    end
    start = sum([dim(spaces(i)) for i = 0:ind-1])
    return start+1:start+dim(spaces(ind))
end

function get_PEPO(pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i = keys(PEPO)]) for dir = 1:4]
    highest_loop = [minimum([i[dir] for i = keys(PEPO)]) for dir = 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = TensorMap(zeros, pspace ⊗ pspace', prod([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj,h,hloop) = zip(conjugated,highest,highest_loop)]))
    for (key, tens) = PEPO
        places = [get_location_PEPO(ind, highest[dir], spaces) for (dir,ind) = enumerate(key)]
        O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end