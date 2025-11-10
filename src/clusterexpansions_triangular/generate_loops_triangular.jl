function update_levels_loops(lattice::Triangular, levels, cycles, m, bonds_indices, levels_sites)
    N = length(unique(vcat(cycles...)))
    if m <= 3 && N <= 5
        for (i, bond) = enumerate(bonds_indices)
            if bond_contained_in_cycles(bond, cycles)
                levels[i] = -1
            end
        end
    else
        error("Number of loops $(m) with N = $(N) not implemented - This corresponds to the cluster $(levels_sites)")
    end
    return levels
end