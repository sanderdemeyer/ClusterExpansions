function make_loop_translationally_invariant_fermionic(A)
    A = (A + permute(A, ((3,4,1,2), (7,8,5,6)))) / 2
    return (A + permute(A, ((4,1,2,3), (8,5,6,7)))) / 2
end

function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = nothing)
    T = scalartype(A)
    if symmetry == "C4"
        A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')        
        A_SW[][:,:,:,:,1,1] = A[]
        A_SE = rotl90_fermionic(A_SW)
        A_NE = rotl90_fermionic(A_SE)
        A_NW = rotl90_fermionic(A_NE)
        As = [A_NW, A_NE, A_SE, A_SW]
    elseif isnothing(symmetry)
        A_NW = zeros(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
        A_NE = zeros(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
        A_SE = zeros(T, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
        A_SW = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
        A_NW[][:,:,1,:,:,1] = A[]
        A_NE[][:,:,1,1,:,:] = A[]
        A_SE[][:,:,:,1,1,:] = A[]
        A_SW[][:,:,:,:,1,1] = A[]
        As = [A_NW, A_NE, A_SE, A_SW]
    else
        error("Symmetry $(symmetry) not implemented")
    end    
    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    As = [As[values[1]], As[values[2]], As[values[3]], As[values[4]]]
    return As
end

function spaces_in_loop(α)
    α >= 4 || throw(ArgumentError("virtual space of the loop cluster must be at least 4"))
    s2 = floor(Int, (α-1)/3)
    s1 = α - 2*s2 - 1
    return [s1, s2]
end

function solve_4_loop_SVD(RHS, space, levels_to_update; verbosity = 0, SVD_truncation = false, entanglement_filtering = true, symmetry = nothing)
    T = scalartype(RHS)
    RHS_rot = permute(RHS, ((4,1,2,3),(8,5,6,7)))
    if norm(RHS - RHS_rot) / norm(RHS) > 1e-36
        if verbosity >= 1
            @warn "Operator is not rotationally invariant. Error = $(norm(RHS - RHS_rot) / norm(RHS)) \n Making the operator rotationally invariant"
        end
        RHS = make_loop_translationally_invariant_fermionic(RHS)
    end
    tensor_norm = norm(RHS)
    RHS /= tensor_norm
    # truncations = [SVD_truncation ? truncdim(s) : notrunc() for s = spaces_in_loop(dim(space))]
    # truncations = [truncdim(s) for s = spaces_in_loop(dim(space))]
    truncations = fill(notrunc(), 2)

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
    ((norm(RHS_reconstruct - RHS) / norm(RHS) > eps(real(T))*1e2) && (verbosity >= 1)) && @warn "Error of SVD in 4-loop = $(norm(RHS_reconstruct - RHS) / norm(RHS))"

    dims = [dim(domain(UU)[1]), dim(domain(UV)[1]), dim(domain(VV)[1])]
    α = 2:dims[1]+1
    β = dims[1]+2:dims[1]+dims[2]+1
    γ = dims[1]+dims[2]+2:dims[1]+dims[2]+dims[3]+1

    vspace = ℂ^(dims[1]+dims[2]+dims[3]+1)
    pspace = ℂ^2
    trivspace = ℂ^1
    A = zeros(T, codomain(UU), vspace ⊗ vspace') 

    A[][:,:,1,α] = UU[] / sqrt(BigFloat(2.0))
    A[][:,:,α,β] = VU[] / sqrt(BigFloat(2.0))
    A[][:,:,β,γ] = UV[] / sqrt(BigFloat(2.0))
    A[][:,:,γ,1] = VV[] / sqrt(BigFloat(2.0))

    error = norm(contract_tensors_symmetric(A) - RHS) / norm(RHS)
    if verbosity >= 1 && error > 1e-2
        @warn "Error in 4-loop before filtering = $(error)"
    elseif verbosity >= 2
        @info "Error in 4-loop before filtering = $(error)"
    end
    if entanglement_filtering
        A = filter_loop(A; maxiter = 10)
    end
    if !(isnothing(space))
        A, error = truncate_loop(A, space)
        if verbosity >= 1 && error > 1e-2
            @warn "Error in 4-loop due to truncation = $(error)"
        elseif verbosity >= 2
            @info "Error in 4-loop due to truncation = $(error)"
        end
    
    end
    vspace = domain(A)[1]
    spaces = i -> (i >= 0) ? spaces(i) : vspace

    error = norm(contract_tensors_symmetric(A) - RHS) / norm(RHS)

    if verbosity >= 1 && error > 1e-2
        @warn "Error in 4-loop = $(error)"
    elseif verbosity >= 2
        @info "Error in 4-loop = $(error)"
    end
    A *= (tensor_norm)^(1/4)
    As = construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = symmetry)
    return As, error, spaces, A
end
