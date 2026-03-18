function get_PEPO(lattice::Honeycomb, T, pspace, PEPO, spaces)
    highest = [maximum([i[dir] for i in keys(PEPO)]) for dir in 1:3]
    highest_loop = [minimum([i[dir] for i in keys(PEPO)]) for dir in 1:3]
    conjugated = Bool[0, 1, 1]
    O = zeros(T, SumSpace(pspace) ⊗ SumSpace(pspace)', ⊗([conj ? (get_sum_space(h, hloop, spaces))' : get_sum_space(h, hloop, spaces) for (conj, h, hloop) in zip(conjugated, highest, highest_loop)]...))
    for (key, tens) in PEPO
        key = [i < 0 ? h - i + 1 : i + 1 for (i, h) in zip(key, highest)]
        O[1, 1, key...] = tens
    end
    return O
end
