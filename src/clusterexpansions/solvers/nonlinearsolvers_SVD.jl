function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]])
    return permute(loop, ((1,2,3,4),(5,6,7,8)))
end

function construct_PEPO_loop(A, pspace, vspace, trivspace)
    A_NW = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
    A_NE = TensorMap(zeros, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
    A_SE = TensorMap(zeros, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
    A_SW = TensorMap(zeros, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
    A_NW[][:,:,1,:,:,1] = A[]
    A_NE[][:,:,1,1,:,:] = A[]
    A_SE[][:,:,:,1,1,:] = A[]
    A_SW[][:,:,:,:,1,1] = A[]
    return [A_NW, A_NE, A_SE, A_SW]
end

function spaces_in_loop(α)
    α >= 4 || throw(ArgumentError("virtual space of the loop cluster must be at least 4"))
    s2 = floor(Int, (α-1)/3)
    s1 = α - 2*s2 - 1
    return [s1, s2]
end

function solve_4_loop(RHS, space, levels_to_update; verbosity = 0)
    println("space = $space")
    truncations = [truncdim(s) for s = spaces_in_loop(dim(space))]

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

    dims = [dim(UU.dom[1]), dim(UV.dom[1]), dim(VV.dom[1])]
    α = 2:dims[1]+1
    β = dims[1]+2:dims[1]+dims[2]+1
    γ = dims[1]+dims[2]+2:dims[1]+dims[2]+dims[3]+1

    Vspace = ℂ^(dims[1]+dims[2]+dims[3]+1)
    pspace = ℂ^2
    trivspace = ℂ^1
    A = TensorMap(zeros, UU.codom, Vspace ⊗ Vspace') 

    A[][:,:,1,α] = UU[] / 4^(1/4)
    A[][:,:,α,β] = VU[] / 4^(1/4)
    A[][:,:,β,γ] = UV[] / 4^(1/4)
    A[][:,:,γ,1] = VV[] / 4^(1/4)

    error = norm(contract_tensors_symmetric(A) - RHS) / norm(RHS)

    if verbosity >= 1 && error > 1e-2
        @warn "Error in 4-loop = $(error)"
    elseif verbosity >= 2
        @info "Error in 4-loop = $(error)"
    end

    As = construct_PEPO_loop(A, pspace, Vspace, trivspace)
    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    As_permuted = [As[values[1]], As[values[2]], As[values[3]], As[values[4]]]
    println("summary = $(summary(As_permuted[1]))")
    return As_permuted, error
end
