abstract type EnvTruncation end

struct ApproximateEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
    maxiter::Int
    tol::Float64
    verbosity::Int
end

function ApproximateEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme;
    check_fidelity::Bool = false, maxiter::Int = 50, tol::Float64 = 1e-10, verbosity::Int = 0)
    ApproximateEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, maxiter, tol, verbosity)
end

function find_isometry(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}} where {E,S<:ElementarySpace},
    trunc_alg::ApproximateEnvTruncation
) 
    T = scalartype(A)
    orig_space_v = domain(A[1])[1] ⊗ domain(A[2])[1]
    trunc_space_v = domain(A[1])[1]
    orig_space_h = domain(A[1])[2] ⊗ domain(A[2])[2]
    trunc_space_h = domain(A[1])[2]
    Ws = get_initial_isometry(T, orig_space_h, orig_space_v, trunc_space_h, trunc_space_v)

    pspace = codomain(A[1])[1]
    M = randn(T, pspace, pspace)

    error = Inf
    expvals = ComplexF64[Inf]
    for i = 1:trunc_alg.maxiter
        t_hor, t_ver, expval = truncation_environment(A, Ws, trunc_alg, M)
        Ws = update_isometry(t_hor, t_ver, trunc_alg.trscheme)
        push!(expvals, expval)

        error = abs(expvals[end] - expvals[end-1])
        if trunc_alg.verbosity > 1
            @info "Step $i: error = $error"
        end
        if error < trunc_alg.tol
            if trunc_alg.verbosity > 1
                @info "Converged after $i iterations: error = $error"
            end
            break
        end
        if i == trunc_alg.maxiter && trunc_alg.verbosity > 0
            if trunc_alg.verbosity > 0
                @warn "Not converged after $i iterations: error = $error"
            end
        end
    end
    return Ws
end

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    Ws::Vector{<:AbstractTensorMap{E,S}},
    trunc_alg::ApproximateEnvTruncation, 
    M::AbstractTensorMap{E,S,1,1}
) where {E,S}
    A_trunc = apply_isometry(A, Ws)
    A_stack = fill(A_trunc, 1, 1, 2)
    A_stack[1, 1, 2] = PEPSKit._dag(A_trunc)
    AAdag = InfinitePEPO(A_stack)

    network = InfiniteSquareNetwork(AAdag)
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    num = PEPSKit.@autoopt @tensor env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2; χ2] * env.corners[2,1,1][χ2; χ3] * 
                                    env.edges[2,1,1][χ3 DE1 DE2; χ4] * env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2; χ6] * 
                                    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2; χ8] * 
                                    twist(A_trunc,2)[Dp1 Dp2; DN1 DE1 DS1 DW1] * conj(A_trunc[Dp3 Dp2; DN2 DE2 DS2 DW2]) * M[Dp3; Dp1]
    denom = PEPSKit.@autoopt @tensor env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2; χ2] * env.corners[2,1,1][χ2; χ3] * 
                                    env.edges[2,1,1][χ3 DE1 DE2; χ4] * env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2; χ6] * 
                                    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2; χ8] * 
                                    twist(A_trunc,2)[Dp1 Dp2; DN1 DE1 DS1 DW1] * conj(A_trunc[Dp1 Dp2; DN2 DE2 DS2 DW2])

    # Horizontal truncation environment
    AL_below = copy(A_trunc)
    AR_below = copy(A_trunc)
    AL_above = apply_isometry(A, Ws, [2])
    AR_above = apply_isometry(A, Ws, [4])
    t_hor = contract_34_patch(AL_above, AR_above, AL_below, AR_below, env)

    # Vertical truncation environment
    AB_below = copy(A_trunc)
    AT_below = copy(A_trunc)
    AB_above = apply_isometry(A, Ws, [1])
    AT_above = apply_isometry(A, Ws, [3])
    t_ver = contract_43_patch(AB_above, AT_above, AB_below, AT_below, env)
    
    return t_hor, t_ver, num / denom
end

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,2,5},
    AR_above::AbstractTensorMap{E,S,2,5},
    AL_below::AbstractTensorMap{E,S,2,4},
    AR_below::AbstractTensorMap{E,S,2,4},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    PEPSKit.@autoopt @tensor t[DCRa DCROa; DCLa DCLOa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa DNLb; χ2] * env.edges[1,1,1][χ2 DNRa DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa DSRb; χ7] * env.edges[3,1,1][χ7 DSLa DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa DWb; χ10] * 
    twist(AL_above, 2)[DpbL DptL; DNLa DCLa DSLa DWa DCLOa] * twist(AR_above, 2)[DpbR DptR; DNRa DEa DSRa DCRa DCROa] * 
    PEPSKit._dag(AL_below)[DptL DpbL; DNLb DCb DSLb DWb] * PEPSKit._dag(AR_below)[DptR DpbR; DNRb DEb DSRb DCb]
    return t
end

function contract_43_patch(
    AB_above::AbstractTensorMap{E,S,2,5},
    AT_above::AbstractTensorMap{E,S,2,5},
    AB_below::AbstractTensorMap{E,S,2,4},
    AT_below::AbstractTensorMap{E,S,2,4},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    PEPSKit.@autoopt @tensor t[DCTa DCTOa; DCBa DCBOa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNa DNb; χ2] * env.corners[2,1,1][χ2; χ3] * 
    env.edges[2,1,1][χ3 DETa DETb; χ4] * env.edges[2,1,1][χ4 DEBa DEBb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSa DSb; χ7] * 
    env.corners[4,1,1][χ7; χ8] * env.edges[4,1,1][χ8 DWBa DWBb; χ9] * env.edges[4,1,1][χ9 DWTa DWTb; χ10] * 
    twist(AB_above, 2)[DpbB DptB; DCBa DEBa DSa DWBa DCBOa] * twist(AT_above, 2)[DpbT DptT; DNa DETa DCTa DWTa DCTOa] * 
    PEPSKit._dag(AB_below)[DptB DpbB; DCb DEBb DSb DWBb] * PEPSKit._dag(AT_below)[DptT DpbT; DNb DETb DCb DWTb]
    return t
end

function update_isometry(t_hor, t_ver, trscheme)
    PE, _, PW = tsvd(t_hor, trunc = trscheme)
    PN, _, PS = tsvd(t_ver, trunc = trscheme)
    return [PN, PE, PS, PW]
end

function approximate_state(
    A::Union{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}},
    trunc_alg::ApproximateEnvTruncation;
    envspace_fidelity = trunc_alg.envspace,
    ctm_alg_fidelity = trunc_alg.ctm_alg
) where {E,S<:ElementarySpace}

    Ws = find_isometry(A, trunc_alg)
    A_trunc = apply_isometry(A, Ws)

    trunc_alg.check_fidelity || return A_trunc, nothing
    overlap = fidelity(A, A_trunc, ctm_alg_fidelity, envspace_fidelity)
    if trunc_alg.verbosity > 1
        @info "Fidelity of approximation is $overlap"
    end
    return A_trunc, overlap
end

