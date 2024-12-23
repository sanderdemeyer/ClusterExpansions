function init_PEPO(pspace::ElementarySpace, trivspace::ElementarySpace)
    return Dict((0,0,0,0) => TensorMap([1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace'))   
end

function init_PEPO(pspace::ElementarySpace, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    A = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
    A[][:,:,1,1,1,1] = exp(-onesite_op)[]
    return Dict((0,0,0,0) => A)
end

function init_PEPO()
    return init_PEPO(ℂ^2, ℂ^1)
end

function init_PEPO(onesite_op::AbstractTensorMap)
    return init_PEPO(ℂ^2, ℂ^1, onesite_op)
end

function get_size_level(highest)
    return sum([2^(2*i) for i = 0:highest])
end

function get_size_level_loop(highest)
    return -10*highest
end

function get_location_PEPO(ind, highest)
    if ind == 0
        return 1
    elseif ind < 0
        return (highest+1-10*(ind+1)):(highest-10*ind)
    end
    h = get_size_level(ind-1)
    return h+1:h+2^(2*ind)
end

function get_PEPO(pspace, PEPO)
    highest = [get_size_level(maximum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    highest_loop = [get_size_level_loop(minimum([i[dir] for i = keys(PEPO)])) for dir = 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = TensorMap(zeros, pspace ⊗ pspace', prod([conj ? (ℂ^(h+hloop))' : ℂ^(h+hloop) for (conj,h,hloop) = zip(conjugated,highest, highest_loop)]))
    for (key, tens) = PEPO
        places = [get_location_PEPO(ind, highest[dir]) for (dir,ind) = enumerate(key)]
        O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end