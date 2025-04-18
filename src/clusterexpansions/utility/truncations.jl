abstract type EnvTruncation end

struct ExactEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
    verbosity::Int
end

struct ApproximateEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
    maxiter::Int
    tol::Float64
    verbosity::Int
end

struct IntermediateEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
    maxiter::Int
    tol::Float64
    verbosity::Int
end

function ExactEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme, check_fidelity::Bool;
    verbosity::Int = 0)
    ExactEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, verbosity)
end

function ApproximateEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme, check_fidelity::Bool;
    maxiter::Int = 50, tol::Float64 = 1e-10, verbosity::Int = 0)
    ApproximateEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, maxiter, tol, verbosity)
end

function IntermediateEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme, check_fidelity::Bool;
    maxiter::Int = 50, tol::Float64 = 1e-10, verbosity::Int = 0)
    IntermediateEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, maxiter, tol, verbosity)
end

function approximate_state(
    A::Union{AbstractTensorMap{E,S,1,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}},
    trunc_alg::EnvTruncation
) where {E,S<:ElementarySpace}
    Ws = find_isometry(A, trunc_alg)
    A_trunc = apply_isometry(A, Ws)

    trunc_alg.check_fidelity || return A_trunc, nothing
    overlap = fidelity(A, A_trunc, trunc_alg.ctm_alg, trunc_alg.envspace)
    if trunc_alg.verbosity > 0
        @info "Fidelity of approximation is $overlap"
    end
    return A_trunc, overlap
end

function approximate_state(
    A::AbstractTensorMap{E,S,2,4},
    trunc_alg::EnvTruncation
) where {E,S}
    T = scalartype(A)
    @tensor ψ[-1; -2 -3 -4 -5] := A[1 2; -2 -3 -4 -5] * isometry(T, fuse(codomain(A)), codomain(A))[-1; 1 2]
    Ws = find_isometry(ψ, trunc_alg)
    A_trunc = apply_isometry(A, Ws)

    trunc_alg.check_fidelity || return A_trunc, nothing
    overlap = fidelity(A, A_trunc, trunc_alg.ctm_alg, trunc_alg.envspace)
    if trunc_alg.verbosity > 0
        @info "Fidelity of approximation is $overlap"
    end
    return A_trunc, overlap
end

function get_trunc_space(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4}} where {E,S<:ElementarySpace}
)
    return domain(A)[1]
end

function get_trunc_space(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}} where {E,S<:ElementarySpace}
)
    return domain(A[1])[1] ⊗ domain(A[2])[1]
end

function find_isometry(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::ExactEnvTruncation
)
    t = truncation_environment(A, trunc_alg)
    # truncspace = get_trunc_space(A)
    Ws = update_isometry(t, trunc_alg.trscheme)#, truncspace
    return Ws
end

function find_isometry(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::Union{ApproximateEnvTruncation, IntermediateEnvTruncation}
) 
    T = scalartype(A)
    space = ℂ^(trunc_alg.trscheme.dim)
    Ws = get_initial_isometry(T, get_trunc_space(A), space, randn)
    error = Inf
    # errors = []
    # fidels = []
    for i = 1:trunc_alg.maxiter
        t = truncation_environment(A, Ws, trunc_alg)
        Ws_new = update_isometry(t, trunc_alg.trscheme)
        _, Σ_new, _ = tsvd(Ws_new[1])
        _, Σ, _ = tsvd(Ws[1])
        error = (norm(Σ_new-Σ)/norm(Σ))
        # error = maximum([norm(Ws_new[dir] - Ws[dir])/norm(Ws[dir]) for dir = 1:4])
        Ws = copy(Ws_new)
        if trunc_alg.verbosity > 1
            @info "Step $i: error = $error"
        end
        # A_trunc = apply_isometry(A, Ws)
        # overlap = fidelity(A, A_trunc, trunc_alg.ctm_alg, trunc_alg.envspace)
        # push!(errors, error)
        # push!(fidels, fidel)
        # println("iter = $i, error = $error, overlap = $fidel")
        if error < trunc_alg.tol
            @info "Converged after $i iterations: error = $error"
            break
        end
        if i == trunc_alg.maxiter && trunc_alg.verbosity > 0
            @warn "Not converged after $i iterations: error = $error"
        end
    end
    # plt = plot(1:length(errors), log.(errors), label="error")
    # display(plt)
    # plt = plot(1:length(fidels), fidels, label="fidel")
    # display(plt)
    return Ws
end

function contract_23_patch(
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,5,1}}
    patch = PEPSKit.@autoopt @tensor t[DNa DNOa; DSa DSOa] := 
        env.corners[1,1,1][χ6; χ1] * env.edges[1,1,1][χ1 DNa DNOa Db DOb; χ2] * env.corners[2,1,1][χ2; χ3] * 
        env.corners[3,1,1][χ3; χ4] * env.edges[3,1,1][χ4 DSa DSOa Db DOb; χ5] * env.corners[4,1,1][χ5; χ6]
    return patch
end

function contract_23_patch(
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,4,1}}
    patch = PEPSKit.@autoopt @tensor t[DNa DNOa; DSa DSOa] := 
        env.corners[1,1,1][χ6; χ1] * env.edges[1,1,1][χ1 DNa DNOa Db; χ2] * env.corners[2,1,1][χ2; χ3] * 
        env.corners[3,1,1][χ3; χ4] * env.edges[3,1,1][χ4 DSa DSOa Db; χ5] * env.corners[4,1,1][χ5; χ6]
    return patch
end

function contract_23_patch(
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DNa; DSa] := 
        env.corners[1,1,1][χ6; χ1] * env.edges[1,1,1][χ1 DNa Db; χ2] * env.corners[2,1,1][χ2; χ3] * 
        env.corners[3,1,1][χ3; χ4] * env.edges[3,1,1][χ4 DSa Db; χ5] * env.corners[4,1,1][χ5; χ6]
    return patch
end

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,1,4},
    AR_above::AbstractTensorMap{E,S,1,4},
    AL_below::AbstractTensorMap{E,S,1,4},
    AR_below::AbstractTensorMap{E,S,1,4},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DCRa; DCLa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa DNLb; χ2] * env.edges[1,1,1][χ2 DNRa DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa DSRb; χ7] * env.edges[3,1,1][χ7 DSLa DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa DWb; χ10] * 
    AL_above[DpL; DNLa DCLa DSLa DWa] * AR_above[DpR; DNRa DEa DSRa DCRa] * 
    conj(AL_below[DpL; DNLb DCb DSLb DWb]) * conj(AR_below[DpR; DNRb DEb DSRb DCb])
    return patch
end

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,1,5},
    AR_above::AbstractTensorMap{E,S,1,5},
    AL_below::AbstractTensorMap{E,S,1,4},
    AR_below::AbstractTensorMap{E,S,1,4},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DCRa DCROa; DCLa DCLOa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa DNLb; χ2] * env.edges[1,1,1][χ2 DNRa DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa DSRb; χ7] * env.edges[3,1,1][χ7 DSLa DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa DWb; χ10] * 
    AL_above[DpL; DNLa DCLa DSLa DWa DCLOa] * AR_above[DpR; DNRa DEa DSRa DCRa DCROa] * 
    conj(AL_below[DpL; DNLb DCb DSLb DWb]) * conj(AR_below[DpR; DNRb DEb DSRb DCb])
    return patch
end

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,1,5},
    AR_above::AbstractTensorMap{E,S,1,5},
    AL_below::AbstractTensorMap{E,S,1,5},
    AR_below::AbstractTensorMap{E,S,1,5},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DCRa DCROa; DCLa DCLOa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa2 DNLb; χ2] * env.edges[1,1,1][χ2 DNRa2 DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa2 DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa2 DSRb; χ7] * env.edges[3,1,1][χ7 DSLa2 DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa2 DWb; χ10] * 
    AL_above[DpL; DNLa1 DCLa  DSLa1 DWa1 DCLOa] * AR_above[DpR; DNRa1 DEa1 DSRa1 DCRa DCROa] * 
    conj(AL_below[DpL; DNLb DCb DSLb DWb DCOb]) * conj(AR_below[DpR; DNRb DEb DSRb DCb DCOb])
    return patch
end

function truncation_environment(
    A::AbstractTensorMap{E,S,1,4},
    trunc_alg::ExactEnvTruncation
) where {E,S}
    network = InfiniteSquareNetwork(InfinitePEPS(A))
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    return contract_23_patch(env)
end

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::ExactEnvTruncation
) where {E,S}
    unitcell = (1,1,1)
    O = InfinitePEPO(A[2], unitcell = unitcell)
    O2 = repeat(O, unitcell[1], unitcell[2], 2*unitcell[3])
    O_perm = permute(A[2], ((1,5,6),(2,3,4)));
    O_conj_unperm = convert(TensorMap, O_perm');
    O_conj = permute(O_conj_unperm, ((1,4),(2,3,5,6)));
    O2[1,1,2] = O_conj
    network = InfiniteSquareNetwork(InfinitePEPS(A[1]), O2)
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    return contract_23_patch(env)
end

function truncation_environment(
    A::AbstractTensorMap{E,S,1,4},
    Ws::Vector{<:AbstractTensorMap{E,S,1,1}},
    trunc_alg::IntermediateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A, Ws)
    network = InfiniteSquareNetwork(InfinitePEPS(A), InfinitePEPS(A_trunc))
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    t =  contract_23_patch(env)
    return t
end

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    Ws::Vector{<:AbstractTensorMap{E,S,2,1}},
    trunc_alg::IntermediateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A..., Ws)
    network = InfiniteSquareNetwork(InfinitePEPS(A[1]), InfinitePEPO(A[2]), InfinitePEPS(A_trunc))
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    return contract_23_patch(env)
end

function truncation_environment(
    A::AbstractTensorMap{E,S,1,4},
    Ws::Vector{<:AbstractTensorMap{E,S,1,1}},
    trunc_alg::ApproximateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A, Ws)
    network = InfiniteSquareNetwork(InfinitePEPS(A_trunc), InfinitePEPS(A_trunc))
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    AL_below = copy(A_trunc)
    AR_below = copy(A_trunc)
    AL_above = apply_isometry(A, Ws, [2])
    AR_above = apply_isometry(A, Ws, [4])
    return contract_34_patch(AL_above, AR_above, AL_below, AR_below,env)
end

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    Ws::Vector{<:AbstractTensorMap{E,S,2,1}},
    trunc_alg::ApproximateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A..., Ws)
    network = InfiniteSquareNetwork(InfinitePEPS(A_trunc), InfinitePEPS(A_trunc))
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    AL_below = copy(A_trunc)
    AR_below = copy(A_trunc)
    AL_above = apply_isometry(A..., Ws, [2])
    AR_above = apply_isometry(A..., Ws, [4])
    return contract_34_patch(AL_above, AR_above, AL_below, AR_below,env)
end
