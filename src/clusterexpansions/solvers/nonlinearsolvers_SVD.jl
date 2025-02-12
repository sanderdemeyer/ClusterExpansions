function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function construct_PEPO_loop(A, pspace, vspace, trivspace)
    T = scalartype(A)
    A_NW = zeros(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
    A_NE = zeros(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
    A_SE = zeros(T, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
    A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
    A_NW[][:,:,1,:,:,1] = A[]
    A_NE[][:,:,1,1,:,:] = A[]
    A_SE[][:,:,:,1,1,:] = A[]
    A_SW[][:,:,:,:,1,1] = A[]
    return [A_NW, A_NE, A_SE, A_SW]
end

function construct_PEPO_loop_symmetric(A_SW, levels_to_update)
    A_SE = rotl90_fermionic(A_SW)
    A_NE = rotl90_fermionic(A_SE)
    A_NW = rotl90_fermionic(A_NE)
    As = [A_NW, A_NE, A_SE, A_SW]

    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    return [As[values[1]], As[values[2]], As[values[3]], As[values[4]]]
end

function spaces_in_loop(α)
    α >= 4 || throw(ArgumentError("virtual space of the loop cluster must be at least 4"))
    s2 = floor(Int, (α-1)/3)
    s1 = α - 2*s2 - 1
    return [s1, s2]
end

function solve_4_loop(RHS, space, levels_to_update; verbosity = 0, filtering = true, symmetry = nothing)
    T = scalartype(RHS)

    RHS_rot = permute(RHS, ((4,1,2,3),(8,5,6,7)))
    if norm(RHS - RHS_rot) / norm(RHS) > 1e-15
        @warn "Operator is not rotationally invariant. Error = $(norm(RHS - RHS_rot) / norm(RHS))"
    end
    tensor_norm = norm(RHS)
    RHS /= tensor_norm
    truncations = [filtering ? notrunc() : truncdim(s) for s = spaces_in_loop(dim(space))]
    # truncations = [truncdim(s) for s = spaces_in_loop(dim(space))]

    U, Σ, V = tsvd(RHS, ((1,2,5,6), (3,4,7,8)), trunc = truncations[1])
    U = U * sqrt(Σ)
    V = sqrt(Σ) * V

    UU, ΣU, VU = tsvd(U, ((1,3), (2,4,5)), trunc = truncations[2])
    UV, ΣV, VV = tsvd(V, ((1,2,4), (3,5)), trunc = truncations[2])

    UU = UU * sqrt(ΣU)
    VU = sqrt(ΣU) * VU

    UV = UV * sqrt(ΣV)
    VV = sqrt(ΣV) * VV

    UV = permute(UV, ((2,3),(1,4)))
    VU = permute(VU, ((2,3),(1,4)))
    VV = permute(VV, ((2,3), (1,)))

    @tensor RHS_reconstruct[-1 -2 -3 -4; -5 -6 -7 -8] := UU[-1 -5; 1] * VU[-2 -6; 1 2] * UV[-3 -7; 2 3] * VV[-4 -8; 3]
    norm(RHS_reconstruct - RHS) / norm(RHS) < 1e-14 || @warn "Error of SVD in 4-loop = $(norm(RHS_reconstruct - RHS) / norm(RHS))"

    dims = [dim(domain(UU)[1]), dim(domain(UV)[1]), dim(domain(VV)[1])]
    α = 2:dims[1]+1
    β = dims[1]+2:dims[1]+dims[2]+1
    γ = dims[1]+dims[2]+2:dims[1]+dims[2]+dims[3]+1

    vspace = ℂ^(dims[1]+dims[2]+dims[3]+1)
    pspace = ℂ^2
    trivspace = ℂ^1
    A = zeros(T, codomain(UU), vspace ⊗ vspace') 

    A[][:,:,1,α] = UU[] / sqrt(2)
    A[][:,:,α,β] = VU[] / sqrt(2)
    A[][:,:,β,γ] = UV[] / sqrt(2)
    A[][:,:,γ,1] = VV[] / sqrt(2)

    println("Norms are $(norm(RHS)) and $(norm(contract_tensors_symmetric(A)))")
    println("Other error is $(norm(contract_tensors_symmetric(A) - RHS_reconstruct) / norm(RHS_reconstruct))")
    error = norm(contract_tensors_symmetric(A) - RHS) / norm(RHS)

    if verbosity >= 1 && error > 1e-2
        @warn "Error in 4-loop before filtering = $(error)"
    elseif verbosity >= 2
        @info "Error in 4-loop before filtering = $(error)"
    end

    if filtering
        A, _ = entanglement_filtering(A; verbosity = verbosity)
        vspace = domain(A)[1]
        spaces = i -> (i >= 0) ? spaces(i) : vspace
    end

    error = norm(contract_tensors_symmetric(A) - RHS) / norm(RHS)

    if verbosity >= 1 && error > 1e-2
        @warn "Error in 4-loop = $(error)"
    elseif verbosity >= 2
        @info "Error in 4-loop = $(error)"
    end

    A *= (tensor_norm)^(1/4)
    if symmetry == "C4"
        A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')        
        A_SW[][:,:,:,:,1,1] = A[]
        As = construct_PEPO_loop_symmetric(A_SW, levels_to_update)
    elseif isnothing(symmetry)
        As = construct_PEPO_loop(A, pspace, vspace, trivspace)
    else
        error("Symmetry $(symmetry) not implemented")
    end
    return As, error, spaces
end
