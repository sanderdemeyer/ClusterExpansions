function init_PEPO(T, pspace::ElementarySpace, trivspace::ElementarySpace)
    return Dict((0, 0, 0, 0) => TensorMap(T[1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace'))
end

function init_PEPO()
    return init_PEPO(T, ℂ^2, ℂ^1)
end

# Very involved way to get the trivial sector
get_length_of_trivspace(sig::T) where {T <: Tuple} = length(sig[1].parameters)
get_length_of_trivspace(sig::Int) = 1

function init_PEPO(T, β, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    exp_H = exp(-β * onesite_op)
    exp_H_perm = permute(exp_H, ((1, 2), ()))
    Isom = permute(isomorphism(T, trivspace' ⊗ trivspace', trivspace' ⊗ trivspace'), ((), (1, 2, 3, 4)))
    A = exp_H_perm * Isom
    return Dict((0, 0, 0, 0) => A)
end

function init_PEPO(T, β, onesite_op::AbstractTensorMap, trivspace)
    I = sectortype(onesite_op)
    return init_PEPO(T, β, trivspace, onesite_op)
end

function get_size_level(highest, highest_loop, spaces)
    return sum([dim(spaces(i)) for i in 0:highest]) + sum([dim(spaces(i)) for i in -1:-1:highest_loop])
end

function get_sum_space(highest, highest_loop, spaces)
    summedspace = ⊕([spaces(i) for i in 0:highest]...)
    # zerospace = ℂ^0
    # total_space = zerospace
    for ind in -1:-1:highest_loop
        summedspace = summedspace ⊕ spaces(ind)
    end
    return summedspace
end

function get_location_PEPO(ind, highest, spaces)
    if ind == 0
        return 1
    elseif ind < 0
        start = sum([dim(spaces(i)) for i in 0:highest]) + sum([dim(spaces(i)) for i in -1:-1:(ind + 1)])
        return (start + 1):(start + dim(spaces(ind)))
    end
    start = sum([dim(spaces(i)) for i in 0:(ind - 1)])
    return (start + 1):(start + dim(spaces(ind)))
end

function get_PEPO(T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i in keys(PEPO)]) for dir in 1:4]
    highest_loop = [minimum([i[dir] for i in keys(PEPO)]) for dir in 1:4]
    conjugated = Bool[0, 0, 1, 1]
    O = zeros(T, SumSpace(pspace) ⊗ SumSpace(pspace)', ⊗([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj, h, hloop) in zip(conjugated, highest, highest_loop)]...))
    for (key, tens) in PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i, h) in zip(key, highest)]
        # places = [get_location_PEPO(ind, highest[dir], spaces) for (dir,ind) = enumerate(key)]
        O[1, 1, key...] = tens
        # O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end
