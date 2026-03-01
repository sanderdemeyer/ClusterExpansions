function init_PEPO(lattice::Triangular, T, pspace::ElementarySpace, trivspace::ElementarySpace)
    return Dict((0, 0, 0, 0, 0, 0) => TensorMap(T[1.0 0.0; 0.0 1.0], pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ trivspace ⊗ trivspace' ⊗ trivspace' ⊗ trivspace'))
end

function init_PEPO(lattice::Triangular, T)
    return init_PEPO(lattice, T, ℂ^2, ℂ^1)
end

function init_PEPO(lattice::Triangular, T, β, trivspace::ElementarySpace, onesite_op::AbstractTensorMap)
    exp_H = exp(-β * onesite_op)
    exp_H_perm = permute(exp_H, ((1, 2), ()))
    Isom = permute(isomorphism(T, trivspace' ⊗ trivspace' ⊗ trivspace', trivspace' ⊗ trivspace' ⊗ trivspace'), ((), (1, 2, 3, 4, 5, 6)))
    A = exp_H_perm * Isom
    return Dict((0, 0, 0, 0, 0, 0) => A)
end

function init_PEPO(lattice::Lattice, T, β, onesite_op::AbstractTensorMap, trivspace)
    I = sectortype(onesite_op)
    return init_PEPO(lattice, T, β, trivspace, onesite_op)
end

function get_PEPO(lattice::Triangular, T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i in keys(PEPO)]) for dir in 1:6]
    highest_loop = [minimum([i[dir] for i in keys(PEPO)]) for dir in 1:6]
    conjugated = Bool[0, 0, 0, 1, 1, 1]
    O = zeros(T, SumSpace(pspace) ⊗ SumSpace(pspace)', ⊗([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj, h, hloop) in zip(conjugated, highest, highest_loop)]...))
    for (key, tens) in PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i, h) in zip(key, highest)]
        # places = [get_location_PEPO(ind, highest[dir], spaces) for (dir,ind) = enumerate(key)]
        O[1, 1, key...] = tens
        # O[][:,:,places[1],places[2],places[3],places[4]] = tens[]
    end
    return O
end
