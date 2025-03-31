function fidelity(A::InfinitePEPS, B::InfinitePEPS, ctm_alg::PEPSKit.CTMRGAlgorithm, envspace)
    env0 = CTMRGEnv(A, envspace);
    envs, = leading_boundary(env0, A, ctm_alg);
    norm_orig = norm(A, envs)

    env0_trunc = CTMRGEnv(B, envspace);
    envs_trunc, = leading_boundary(env0_trunc, B, ctm_alg);
    norm_trunc = norm(B, envs_trunc)

    network = InfiniteSquareNetwork(A, B)
    env0_netw = CTMRGEnv(network, envspace)
    envs_netw, = leading_boundary(env0_netw, network, ctm_alg)
    overlap = network_value(network, envs_netw)
    
    return abs(overlap / sqrt(norm_orig*norm_trunc))
end

function fidelity(A::AbstractTensorMap{E,S,1,4}, B::AbstractTensorMap{E,S,1,4}, ctm_alg, envspace) where {E,S}
    return fidelity(InfinitePEPS(A), InfinitePEPS(B), ctm_alg, envspace)
end

function fidelity(A::AbstractTensorMap{E,S,2,4}, B::AbstractTensorMap{E,S,2,4}, ctm_alg, envspace) where {E,S}
    @assert scalartype(A) == scalartype(B)
    T = scarlartype(A)
    @tensor A_fused[-1; -2 -3 -4 -5] := A[1 2; -2 -3 -4 -5] * isometry(T, fuse(codomain(A)), codomain(A))[-1; 1 2]
    @tensor B_fused[-1; -2 -3 -4 -5] := B[1 2; -2 -3 -4 -5] * isometry(T, fuse(codomain(B)), codomain(B))[-1; 1 2]

    return fidelity(InfinitePEPS(A_fused), InfinitePEPS(B_fused), ctm_alg, envspace)
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * Ws[1][2 3; -2] * Ws[2][4 5; -3] * Ws[3][6 7; -4] * Ws[4][8 9; -5]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, W::AbstractTensorMap{E,S,2,1}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * W[2 3; -2] * W[4 5; -3] * W[6 7; -4] * W[8 9; -5]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,2,4}, O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    @tensor A_trunc[-1 -2; -3 -4 -5 -6] := A[1 -2; 2 4 6 8] * O[-1 1; 3 5 7 9] * Ws[1][2 3; -3] * Ws[2][4 5; -4] * Ws[3][6 7; -5] * Ws[4][8 9; -6]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,2,4}, O::AbstractTensorMap{E,S,2,4}, W::AbstractTensorMap{E,S,2,1}) where {E,S}
    @tensor A_trunc[-1 -2; -3 -4 -5 -6] := A[1 -2; 2 4 6 8] * O[-1 1; 3 5 7 9] * W[2 3; -3] * W[4 5; -4] * W[6 7; -5] * W[8 9; -6]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, Ws::Vector{<:AbstractTensorMap{E,S,1,1}}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[-1; 1 2 3 4] * Ws[1][1; -2] * Ws[2][2; -3] * Ws[3][3; -4] * Ws[4][4; -5]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, W::AbstractTensorMap{E,S,1,1}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[-1; 1 2 3 4] * W[1; -2] * W[2; -3] * W[3; -4] * W[4; -5]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,1,1}}) where {E,S}
    @tensor A_trunc[-1 -2; -3 -4 -5 -6] := A[-1 -2; 1 2 3 4] * Ws[1][1; -3] * Ws[2][2; -4] * Ws[3][3; -5] * Ws[4][4; -6]
    return A_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,2,4}, W::AbstractTensorMap{E,S,1,1}) where {E,S}
    @tensor A_trunc[-1 -2; -3 -4 -5 -6] := A[-1 -2; 1 2 3 4] * W[1; -3] * W[2; -4] * W[3; -5] * W[4; -6]
    return A_trunc
end

function find_isometry_fullenv(
    ψ::AbstractTensorMap{E,S,1,4},
    O::AbstractTensorMap{E,S,2,4},
    space,
    ctm_alg::PEPSKit.CTMRGAlgorithm,
    envspace
    ) where {E,S}
    T = scalartype(ψ)
    O_conj = O'
    unitcell = [1,1,1]
    O = InfinitePEPO(O; unitcell = Tuple(unitcell))
    O2 = repeat(O, unitcell[1], unitcell[2], 2*unitcell[3])
    # O2[1,1,2] = O_conj

    # n = InfiniteSquareNetwork(ψ, O)
    network = InfiniteSquareNetwork(InfinitePEPS(ψ; unitcell = Tuple(unitcell[1:2])), O2)

    env, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)

    PEPSKit.@autoopt @tensor t[DNa DNOa; DSa DSOa] := 
        env.corners[1,1,1][χ6; χ1] * env.edges[1,1,1][χ1 DNa DNOa Db DOb; χ2] * env.corners[2,1,1][χ2; χ3] * 
        env.corners[3,1,1][χ3; χ4] * env.edges[3,1,1][χ4 DSa DSOa Db DOb; χ5] * env.corners[4,1,1][χ5; χ6]

    _, _, V = tsvd(t, trunc = truncspace(space))

    Ws = [i > 2 ? zeros(T, domain(ψ)[i] ⊗ domain(O[])[i], space') : zeros(T, domain(ψ)[i] ⊗ domain(O[])[i], space) for i = 1:4]
    for dir in 1:4
        Ws[dir][][:, :, :] = reshape(V[][:, :, :], (dim(domain(ψ)[dir]), dim(domain(O[])[dir]), dim(space)))
    end
    return Ws
end

function find_isometry_fullenv(
    ψ::AbstractTensorMap{E,S,1,4}, 
    space,
    ctm_alg::PEPSKit.CTMRGAlgorithm, 
    envspace
    ) where {E,S}
    T = scalartype(ψ)

    peps = InfinitePEPS(ψ)
    env, = leading_boundary(CTMRGEnv(peps, envspace), peps, ctm_alg)

    PEPSKit.@autoopt @tensor t[DNa; DSa] := 
        env.corners[1,1,1][χ6; χ1] * env.edges[1,1,1][χ1 DNa Db; χ2] * env.corners[2,1,1][χ2; χ3] * 
        env.corners[3,1,1][χ3; χ4] * env.edges[3,1,1][χ4 DSa Db; χ5] * env.corners[4,1,1][χ5; χ6]

    _, _, V = tsvd(t, trunc = truncspace(space))

    Ws = [i > 2 ? zeros(T, domain(ψ)[i], space') : zeros(T, domain(ψ)[i], space) for i = 1:4]
    for dir in 1:4
        Ws[dir][][:, :, :] = reshape(V[][:, :, :], (dim(domain(ψ)[dir]), dim(space)))
    end
    return Ws
end

function find_isometry_fullenv(
    O::AbstractTensorMap{E,S,2,4},
    space,
    ctm_alg::PEPSKit.CTMRGAlgorithm,
    envspace
    ) where {E,S}

    T = scalartype(O)
    @tensor ψ[-1; -2 -3 -4 -5] := O[1 2; -2 -3 -4 -5] * isometry(T, fuse(codomain(O)), codomain(O))[-1; 1 2]
    return find_isometry_fullenv(ψ, space, ctm_alg, envspace)
end

function find_isometry_approx(
    O::AbstractTensorMap{E,S,2,4},
    space,
    ctm_alg::PEPSKit.CTMRGAlgorithm,
    envspace;
    maxiter = 10,
    tol = 1e-5, 
    method = "approx"
    ) where {E,S}

    T = scalartype(O)
    @tensor ψ[-1; -2 -3 -4 -5] := O[1 2; -2 -3 -4 -5] * isometry(T, fuse(codomain(O)), codomain(O))[-1; 1 2]
    return find_isometry_approx(ψ, space, ctm_alg, envspace; maxiter, tol, method)
end

function contract_24_patch(network, env, Ws)
    ψ = network[1,1][1]
    ψ_trunc = network[1,1][2]
    PEPSKit.@autoopt @tensor t[DCLa; DCRa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa2 DNLb; χ2] * env.edges[1,1,1][χ2 DNRa2 DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa2 DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa2 DSRb; χ7] * env.edges[3,1,1][χ7 DSLa2 DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa2 DWb; χ10] * 
    ψ[DpL; DNLa1 DCLa DSLa1 DWa1] * ψ[DpR; DNRa1 DEa1 DSRa1 DCRa] * 
    conj(ψ_trunc[DpL; DNLb DCb DSLb DWb]) * conj(ψ_trunc[DpR; DNRb DEb DSRb DCb]) * 
    Ws[1][DNLa1; DNLa2] * Ws[3][DSLa1; DSLa2] * Ws[4][DWa1; DWa2] * 
    Ws[1][DNRa1; DNRa2] * Ws[2][DEa1; DEa2] * Ws[3][DSRa1; DSRa2]
    return t
end

function update_isometry(t, spaces, space_trunc)
    _, _, V = tsvd(t, trunc = truncspace(space_trunc))
    Ws_new = [dir > 2 ? zeros(T, spaces[dir], space_trunc') : zeros(T, spaces[dir], space_trunc) for dir = 1:4]
    for dir in 1:4
        Ws_new[dir][][:, :, :] = reshape(V[][:, :, :], (dim(spaces[dir]), dim(space_trunc)))
    end
    return Ws_new
end

function find_isometry_approx(
    ψ::AbstractTensorMap{E,S,1,4},
    space,
    ctm_alg::PEPSKit.CTMRGAlgorithm, 
    envspace;
    maxiter = 10,
    tol = 1e-5, 
    method = "approx"
    ) where {E,S}

    Ws = [i > 2 ? randn(T, domain(ψ)[i], space') : randn(T, domain(ψ)[i], space) for i = 1:4]
    error = Inf
    fidel = Inf
    for i = 1:maxiter
        println("In iteration $i, the error is $error and fidelity = $fidel")

        ψ_trunc = apply_isometry(ψ, Ws)
        if method == "approx"
            network_env = InfiniteSquareNetwork(InfinitePEPS(ψ_trunc), InfinitePEPS(ψ_trunc))
            network = InfiniteSquareNetwork(InfinitePEPS(ψ), InfinitePEPS(ψ_trunc))
        elseif method == "intermediate"
            network_env = InfiniteSquareNetwork(InfinitePEPS(ψ_trunc), InfinitePEPS(ψ))
            network = InfiniteSquareNetwork(InfinitePEPS(ψ), InfinitePEPS(ψ))
        else
            @error "Method $(method) not implemented"
        end
        env, = leading_boundary(CTMRGEnv(network_env, envspace), network_env, ctm_alg)

        t = contract_24_patch(network, env, Ws)
        Ws_new = update_isometry(t, domain(ψ), space)

        error = maximum([norm(Ws_new[dir] - Ws[dir])/norm(Ws[dir]) for dir = 1:4])
        ψ_trunc_new = apply_isometry(ψ, Ws_new)
        fidel = fidelity(ψ_trunc_new, ψ_trunc, ctm_alg, envspace) 
        Ws = copy(Ws_new)
        if error < tol
            break
        end
    end
    println("Converged isometry gives error $error and fidelity $fidel")
    return Ws
end

# function find_isometry_intermediate(
#     ψ::AbstractTensorMap{E,S,1,4},
#     space,
#     ctm_alg::PEPSKit.CTMRGAlgorithm, 
#     envspace;
#     maxiter = 10,
#     tol = 1e-5
#     ) where {E,S}

#     Ws = [i > 2 ? randn(T, domain(ψ)[i], space') : randn(T, domain(ψ)[i], space) for i = 1:4]
#     error = Inf
#     fidel = Inf
#     for i = 1:maxiter
#         println("In iteration $i, the error is $error and fidelity = $fidel")

#         ψ_trunc = apply_isometry(ψ, Ws)
#         network = InfiniteSquareNetwork(InfinitePEPS(ψ_trunc), InfinitePEPS(ψ))
#         env, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)

#         t = contract_24_patch(ψ, ψ, env)
#         Ws_new = update_isometry(t, domain(ψ), space)

#         error = maximum([norm(Ws_new[dir] - Ws[dir])/norm(Ws[dir]) for dir = 1:4])
#         ψ_trunc_new = apply_isometry(ψ, Ws_new)
#         fidel = fidelity(ψ_trunc_new, ψ_trunc, ctm_alg, envspace) 
#         Ws = copy(Ws_new)
#         if error < tol
#             break
#         end

#     end
#     println("Converged isometry gives error $error and fidelity $fidel")
#     return Ws
# end

function apply_PEPO_fullenv(
    ψ::AbstractTensorMap{E,S,1,4},
    O::AbstractTensorMap{E,S,2,4},
    ctm_alg,
    envspace;
    space = domain(ψ)[1],
    method = "fullenv",
    check_fidelity = false    
) where {E,S}
    return approximate_fullenv([ψ, O], space, ctm_alg, envspace; method = method, check_fidelity = check_fidelity)
end

# function approximate_fullenv(
#     ψ::AbstractTensorMap{E,S,1,4},
#     O::AbstractTensorMap{E,S,2,4},
#     space,
#     ctm_alg,
#     envspace;
#     method = "fullenv",
#     check_fidelity = false
# ) where {E,S}
#     if method == "fullenv"
#         Ws = find_isometry_fullenv(ψ, O, space, ctm_alg, envspace)
#     else
#         @error "Method $(method) not implemented"
#     end
#     println(typeof(Ws))
#     ψ_trunc = apply_isometry(ψ, O, Ws)
#     overlap = check_fidelity ? fidelity(ψ, ψ_trunc, ctm_alg, envspace) : nothing
#     return ψ_trunc, overlap
# end

# function approximate_fullenv(
#     A::Union{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
#     space,
#     ctm_alg,
#     envspace;
#     method = "fullenv",
#     check_fidelity = false
# ) where {E,S}
#     space == codomain(A)[1] && return ψ
#     if method == "fullenv"
#         Ws = find_isometry_fullenv(A, space, ctm_alg, envspace)
#     else
#         @error "Method $(method) not implemented"
#     end
#     println(typeof(Ws))
#     A_trunc = apply_isometry(A, Ws)
#     overlap = check_fidelity ? fidelity(A, A_trunc, ctm_alg, envspace) : nothing
#     return A_trunc, overlap
# end

function approximate_fullenv(
    A::Vector,
    space,
    ctm_alg,
    envspace;
    method = "fullenv",
    check_fidelity = false
)
    if method == "fullenv"
        Ws = find_isometry_fullenv(A..., space, ctm_alg, envspace)
    else
        @error "Method $(method) not implemented"
    end
    A_trunc = apply_isometry(A..., Ws)
    overlap = check_fidelity ? fidelity(A[1], A_trunc, ctm_alg, envspace) : nothing
    return A_trunc, overlap
end

function apply_PEPO_exact(
    ψ::Union{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    O::AbstractTensorMap{E,S,2,4},
) where {E,S}
    T = scalartype(ψ)
    # Not sure whether this works for fermions
    Ws = [dir > 2 ? isometry(T, domain(ψ)[dir] ⊗ domain(O)[dir], fuse(domain(ψ)[dir], domain(O)[dir])') : isometry(T, domain(ψ)[dir] ⊗ domain(O)[dir], fuse(domain(ψ)[dir], domain(O)[dir])) for dir = 1:4]
    return apply_isometry(ψ, O, Ws)
end
