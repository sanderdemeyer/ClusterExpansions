function init_PEPO(T, pspace::ElementarySpace, trivspace::ElementarySpace)
    return Dict((0,0,0,0) => TensorMap(T[1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace'))   
end

function init_PEPO()
    return init_PEPO(T, ℂ^2, ℂ^1)
end

function init_PEPO(T, β, pspace::ElementarySpace, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    A = zeros(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace')
    I = sectortype(A)
    if I == Trivial
        trivsector = I()
    else
        trivsector = I(0)
    end
    block(A, trivsector) .= exp(-β*onesite_op).data
    return Dict((0,0,0,0) => A)
end

function init_PEPO(T, β, onesite_op::AbstractTensorMap, trivspace)
    I = sectortype(onesite_op)
    return init_PEPO(T, β, domain(onesite_op)[1], trivspace, onesite_op)
end

function get_size_level(highest, highest_loop, spaces)
    return sum([dim(spaces(i)) for i = 0:highest]) + sum([dim(spaces(i)) for i = -1:-1:highest_loop])
end

function get_sum_space(highest, highest_loop, spaces)
    summedspace = ⊕([spaces(i) for i = 0:highest]...)
    # zerospace = ℂ^0
    # total_space = zerospace
    for ind = -1:-1:highest_loop
        summedspace = summedspace ⊕ spaces(ind)
    end
    return summedspace
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

function get_PEPO(T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i = keys(PEPO)]) for dir = 1:4]
    highest_loop = [minimum([i[dir] for i = keys(PEPO)]) for dir = 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = zeros(T, SumSpace(pspace) ⊗ SumSpace(pspace)', ⊗([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj,h,hloop) = zip(conjugated,highest,highest_loop)]...))
    for (key, tens) = PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i,h) = zip(key,highest)]
        # places = [get_location_PEPO(ind, highest[dir], spaces) for (dir,ind) = enumerate(key)]
        O[1,1,key...] = tens
        # O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end