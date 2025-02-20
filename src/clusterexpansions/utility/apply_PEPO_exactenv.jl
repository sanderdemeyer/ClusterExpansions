function apply_isometry(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * Ws[1][2 3; -2] * Ws[2][4 5; -3] * Ws[3][6 7; -4] * Ws[4][8 9; -5]
    return A_trunc
end

function apply_isometry_general(A, O, Ws)
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * Ws[1][2 3; -2] * Ws[2][4 5; -3] * Ws[3][6 7; -4] * Ws[4][8 9; -5]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, W::AbstractTensorMap{E,S,2,1}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * W[2 3; -2] * W[4 5; -3] * W[6 7; -4] * W[8 9; -5]
    return A_trunc
end

function find_isometry_exact(ψ, O, space)
    T = scalartype(ψ)
    O_conj = O'
    unitcell = [1,1,1]
    O = InfinitePEPO(O; unitcell = Tuple(unitcell))
    O2 = repeat(O, unitcell[1], unitcell[2], 2*unitcell[3])
    # O2[1,1,2] = O_conj

    # n = InfiniteSquareNetwork(ψ, O)
    network = InfiniteSquareNetwork(InfinitePEPS(ψ; unitcell = Tuple(unitcell[1:2])), O2)

    ctm_alg = SimultaneousCTMRG(;
    tol=1e-10,
    miniter=4,
    maxiter=100,
    verbosity=0,
    svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
    )

    env, = leading_boundary(CTMRGEnv(network, space), network, ctm_alg)
   
    PEPSKit.@autoopt @tensor t[DNa DNOa; DSa DSOa] := 
        env.corners[1,1,1][χ6; χ1] * env.edges[1,1,1][χ1 DNa DNOa Db DOb; χ2] * env.corners[2,1,1][χ2; χ3] * 
        env.corners[3,1,1][χ3; χ4] * env.edges[3,1,1][χ4 DSa DSOa Db DOb; χ5] * env.corners[4,1,1][χ5; χ6]

    U, Σ, V = tsvd(t, trunc = truncspace(space))

    Ws = [i > 2 ? zeros(T, domain(ψ)[i] ⊗ domain(O[])[i], space') : zeros(T, domain(ψ)[i] ⊗ domain(O[])[i], space) for i = 1:4]
    for dir in 1:4
        Ws[dir][][:, :, :] = reshape(V[][:, :, :], (dim(domain(ψ)[dir]), dim(domain(O[])[dir]), dim(space)))
    end
    return Ws
end

function apply_PEPO(
    ψ::AbstractTensorMap{E,S,1,4},
    O::AbstractTensorMap{E,S,2,4};
    space = domain(ψ)[1],
    method = "exact"
) where {E,S}
    T = scalartype(ψ)
    if method == "exact"
        Ws = find_isometry_exact(ψ, O, space)
    else
        @error "Method $(method) not implemented"
    end
    return apply_isometry(ψ, O, Ws)
end