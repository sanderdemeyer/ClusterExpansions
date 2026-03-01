# This is not valid in the fermionic case. For fermions, flippers have to be used, which are defined in TensorKit v0.14

function rotl90_fermionic(A::AbstractTensorMap{E, S, 1, 4}) where {E, S <: ElementarySpace}
    return twist(flip(rotl90(A), [3 5]), [3 5])
    A = rotl90(A)

    I₂ = isometry(A.dom[2], (A.dom[2])')
    I₄ = isometry(A.dom[4], (A.dom[4])')

    @tensor A_rot[-1; -2 -3 -4 -5] := A[-1; -2 1 -4 2] * I₂[1; -3] * I₄[2; -5]
    return A_rot
end

function rotl90_fermionic(A::AbstractTensorMap{E, S, 2, 4}) where {E, S <: ElementarySpace}
    return twist(flip(rotl90(A), [4 6]), [4 6])
    A = rotl90(A)

    I₂ = isometry(A.dom[2], (A.dom[2])')
    I₄ = isometry(A.dom[4], (A.dom[4])')

    @tensor A_rot[-1 -2; -3 -4 -5 -6] := A[-1 -2; -3 1 -5 2] * I₂[1; -4] * I₄[2; -6]
    return A_rot
end

function rotl180_fermionic(A::AbstractTensorMap{E, S, 1, 4}) where {E, S <: ElementarySpace}
    return twist(flip(rotl90(rotl90(A)), [2 3 4 5]), [2 3 4 5])
    A = rotl90(rotl90(A))

    I₁ = isometry(A.dom[1], (A.dom[1])')
    I₂ = isometry(A.dom[2], (A.dom[2])')
    I₃ = isometry(A.dom[3], (A.dom[3])')
    I₄ = isometry(A.dom[4], (A.dom[4])')
    @tensor A_rot[-1; -2 -3 -4 -5] := A[-1; 1 2 3 4] * I₁[1; -2] * I₂[2; -3] * I₃[3; -4] * I₄[4; -5]
    return A_rot
end

function rotl180_fermionic(A::AbstractTensorMap{E, S, 2, 4}) where {E, S <: ElementarySpace}
    return twist(flip(rotl90(rotl90(A)), [3 4 5 6]), [3 4 5 6])
    A = rotl90(rotl90(A))

    I₁ = isometry(A.dom[1], (A.dom[1])')
    I₂ = isometry(A.dom[2], (A.dom[2])')
    I₃ = isometry(A.dom[3], (A.dom[3])')
    I₄ = isometry(A.dom[4], (A.dom[4])')
    @tensor A_rot[-1 -2; -3 -4 -5 -6] := A[-1 -2; 1 2 3 4] * I₁[1; -3] * I₂[2; -4] * I₃[3; -5] * I₄[4; -6]
    return A_rot
end

function symmetrize(symmetry, levels_to_update, solutions; N = length(levels_to_update))
    @assert length(solutions) == N "Number of solutions $(length(solutions)) does not match number of levels to update $(N)"
    if isnothing(symmetry)
        return levels_to_update, solutions
    elseif symmetry == "C4"
        return symmetrize_C4(levels_to_update, solutions, N)
    else
        @warn "Symmetry $(symmetry) is not implemented"
        return levels_to_update, solutions
    end
end

function symmetrize_C4(levels_to_update, solutions, N)
    if N == 1
        # for _ = 1:2
        #     new_levels = tuple(circshift(collect(levels_to_update[end]), -1)...)
        #     new_solution = rotl90_fermionic(solutions[end])
        #     push!(levels_to_update, new_levels)
        #     push!(solutions, new_solution)
        # end
        new_levels = tuple(circshift(collect(levels_to_update[1]), -1)...)
        new_solution = rotl90_fermionic(solutions[1])
        push!(levels_to_update, new_levels)
        push!(solutions, new_solution)
    elseif N == 2
        for i in 1:2
            new_levels = tuple(circshift(collect(levels_to_update[i]), -1)...)
            new_solution = rotl90_fermionic(solutions[i])
            push!(levels_to_update, new_levels)
            push!(solutions, new_solution)
        end
    end
    return levels_to_update, solutions
end
