function rotl90_fermionic(A::AbstractTensorMap{E,S,1,4}) where {E,S<:ElementarySpace}
    return twist(flip(permute(A, ((1,),(3,4,5,2))), [3 5]), [3 5])
end

function rotl90_fermionic(A::AbstractTensorMap{E,S,2,4}) where {E,S<:ElementarySpace}
    return twist(flip(permute(A, ((1,2),(4,5,6,3))), [4 6]), [4 6])
end

function rotl180_fermionic(A::AbstractTensorMap{E,S,1,4}) where {E,S<:ElementarySpace}
    return rotl90_fermionic(rotl90_fermionic(A))
end

function rotl180_fermionic(A::AbstractTensorMap{E,S,2,4}) where {E,S<:ElementarySpace}
    return rotl90_fermionic(rotl90_fermionic(A))
end

function symmetrize(symmetry, levels_to_update, solutions; N = length(levels_to_update))
    @assert length(solutions) == N "Number of solutions $(length(solutions)) does not match number of levels to update $(N)"
    if isnothing(symmetry)
        return levels_to_update, solutions
    elseif symmetry == "C4v"
        return symmetrize_C4(levels_to_update, solutions, N)
    else
        @error "Symmetry $(symmetry) is not implemented"
        return levels_to_update, solutions
    end
end

function symmetrize_C4(levels_to_update, solutions, N)
    rotate_i_times(n) = x -> foldl(|>, fill(rotl90_fermionic, n), init = x)

    new_levels = copy(levels_to_update)
    new_solutions = copy(solutions)
    for i = 1:3
        for (level_to_update,solution) = zip(levels_to_update,solutions)
            new_level = tuple(circshift(collect(level_to_update), -i)...)
            if new_level ∉ new_levels
                new_solution = rotate_i_times(i)(solution)
                push!(new_levels, new_level)
                push!(new_solutions, new_solution)
            end
        end
    end
    return new_levels, new_solutions
end