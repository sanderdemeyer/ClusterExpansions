# function rotl90_fermionic(A::AbstractTensorMap{E,S,1,4}) where {E,S<:ElementarySpace}
#     return twist(flip(rotl90(A), [3 5]), [3 5])
# end

function rotl60_fermionic(A::AbstractTensorMap{E,S,2,6}) where {E,S<:ElementarySpace}
    A_rot = permute(A, ((1,2),(4,5,6,7,8,3)))
    return twist(flip(A_rot, [5 8]), [5 8])
end

function symmetrize(lattice::Triangular, symmetry, levels_to_update, solutions; N = length(levels_to_update))
    @assert length(solutions) == N "Number of solutions $(length(solutions)) does not match number of levels to update $(N)"
    if isnothing(symmetry)
        return levels_to_update, solutions
    elseif symmetry == "C6"
        return symmetrize_C6(levels_to_update, solutions, N)
    else
        @error "Symmetry $(symmetry) is not implemented for $(lattice)"
        return levels_to_update, solutions
    end
end

function symmetrize_C6(levels_to_update, solutions, N)
    if N == 1
        for i = 1:6
            new_levels = tuple(circshift(collect(levels_to_update[i]), -1)...)
            new_solution = rotl60_fermionic(solutions[i])
            push!(levels_to_update, new_levels)
            push!(solutions, new_solution)
        end
    elseif N == 2
        for i = 0:2
            new_levels1 = tuple(circshift(collect(levels_to_update[2*i+1]), -1)...)
            new_levels2 = tuple(circshift(collect(levels_to_update[2*i+2]), -1)...)
            new_solution1 = rotl60_fermionic(solutions[2*i+1])
            new_solution2 = rotl60_fermionic(solutions[2*i+2])
            push!(levels_to_update, new_levels1)
            push!(solutions, new_solution1)
            push!(levels_to_update, new_levels2)
            push!(solutions, new_solution2)
        end
    elseif N == 3
        for i = 1:3
            new_levels = tuple(circshift(collect(levels_to_update[i]), -3)...)
            new_solution = rotl60_fermionic(rotl60_fermionic(rotl60_fermionic(solutions[i])))
            push!(levels_to_update, new_levels)
            push!(solutions, new_solution)
        end
    end
    return levels_to_update, solutions
end

# ce_alg_c4 = spinless_fermion_operators(1.0, 0.0, 0.0; symmetry = "C4");
# ce_alg = spinless_fermion_operators(1.0, 0.0, 0.0; symmetry = nothing);

# O_c4 = evolution_operator(ce_alg_c4, 0.2);
# O = evolution_operator(ce_alg, 0.2);
# ϵ =  norm(O - O_c4)