function iscorner(site₁, site₂, site₃)
    dir12 = [site₂[1] - site₁[1], site₂[2] - site₁[2]]
    dir23 = [site₃[1] - site₂[1], site₃[2] - site₂[2]]
    return dot(dir12, dir23) == 0
end

function bond_contained_in_cycle(bond, cycle)
    return (bond[1] ∈ cycle) && (bond[2] ∈ cycle)
end

function bond_contained_in_cycles(bond, cycles)
    for cycle in cycles
        if bond_contained_in_cycle(bond, cycle)
            return true
        end
    end
    return false
end

function update_levels_loops(levels, cycles, m, bonds_indices, levels_sites)
    N = length(unique(vcat(cycles...)))
    if m == 1
        if N == 4 || N == 8
            for (i, bond) in enumerate(bonds_indices)
                if bond_contained_in_cycle(bond, cycles[1])
                    levels[i] = -1
                end
            end
        elseif N == 10
            corners = [i for i in cycles[1] if iscorner(levels_sites[mod1(i - 1, N)], levels_sites[i], levels_sites[mod1(i + 1, N)])]
            for (i, bond) in enumerate(bonds_indices)
                if bond_contained_in_cycle(bond, cycles[1])
                    if (bonds_indices[i - 1][1] ∈ corners) && (bonds_indices[i + 1][2] ∈ corners)
                        levels[i] = -2
                    else
                        levels[i] = -1
                    end
                end
            end
        else
            error("Number of loops $(m) with N = $(N) not implemented - This corresponds to the cluster $(levels_sites)")
        end
    elseif m == 2
        if N == 6 || N == 7
            for (i, bond) in enumerate(bonds_indices)
                if bond_contained_in_cycle(bond, vcat(cycles...))
                    levels[i] = -1
                end
            end
        elseif N == 8
            for (i, (u, v)) in enumerate(bonds_indices)
                if bond_contained_in_cycle((u, v), vcat(cycles...))
                    if ((u ∈ cycles[1]) && (v ∈ cycles[2])) || ((u ∈ cycles[2]) && (v ∈ cycles[1]))
                        levels[i] = -2
                    else
                        levels[i] = -1
                    end
                end
            end
        else
            error("Number of loops $(m) with N = $(N) not implemented - This corresponds to the cluster $(levels_sites)")
        end
    elseif m == 3
        if N == 8
            for (i, bond) in enumerate(bonds_indices)
                if bond_contained_in_cycle(bond, vcat(cycles...))
                    if sum([bond_contained_in_cycle(bond, cycle) for cycle in cycles]) == 2
                        levels[i] = -2
                    else
                        levels[i] = -1
                    end
                end
            end
        elseif N == 9
            for (i, (u, v)) in enumerate(bonds_indices)
                if bond_contained_in_cycle((u, v), vcat(cycles...))
                    if ((u ∈ cycles[1]) && (v ∈ cycles[2])) || ((u ∈ cycles[2]) && (v ∈ cycles[1]))
                        levels[i] = -2
                    else
                        levels[i] = -1
                    end
                end
            end
        else
            error("Number of loops $(m) with N = $(N) not implemented - This corresponds to the cluster $(levels_sites)")
        end
    elseif m == 4 && N == 9
        for (i, bond) in enumerate(bonds_indices)
            if bond_contained_in_cycle(bond, vcat(cycles...))
                if sum([bond_contained_in_cycle(bond, cycle) for cycle in cycles]) == 2
                    levels[i] = -2
                else
                    levels[i] = -1
                end
            end
        end
    else
        error("Number of loops $(m) with N = $(N) not implemented - This corresponds to the cluster $(levels_sites)")
    end
    return levels
end
