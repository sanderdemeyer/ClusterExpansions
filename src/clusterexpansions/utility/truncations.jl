abstract type EnvTruncation end

struct ExactEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
end

struct ApproximateEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
    maxiter::Int
    tol::Float64
end

struct IntermediateEnvTruncation <: EnvTruncation
    ctm_alg
    envspace::ElementarySpace
    trscheme::TruncationScheme
    check_fidelity::Bool
    maxiter::Int
    tol::Float64
end

function ApproximateEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme, check_fidelity::Bool;
    maxiter::Int = 100, tol::Float64 = 1e-6)
    new(ctm_alg, envspace, trscheme, check_fidelity, maxiter, tol)
end

function approximate(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::EnvTruncation
)
    Ws = find_isometry(A, trunc_alg)
    A_trunc = apply_isometry(A, Ws)
    overlap = check_fidelity ? fidelity(A, A_trunc, trunc_alg.ctm_alg, trunc_alg.envspace) : nothing
    return A_trunc, overlap
end

function find_isometry(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::ExactEnvTruncation
)
    t = truncation_environment(A, trunc_alg)
    Ws = update_isometry(t, alg.trscheme, domain(A))
    return Ws
end

function find_isometry(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::Union{ApproximateEnvTruncation, IntermediateEnvTruncation}
) 
    Ws = [i > 2 ? randn(T, domain(ψ)[i], space') : randn(T, domain(ψ)[i], space) for i = 1:4]
    error = Inf
    for i = 1:trunc_alg.maxiter
        A_trunc = apply_isometry(A, Ws)
        t = truncation_environment(A, A_trunc, trunc_alg)
        Ws_new = update_isometry(t, alg.trscheme, domain(A))
        error = maximum([norm(Ws_new[dir] - Ws[dir])/norm(Ws[dir]) for dir = 1:4])
        Ws = copy(Ws_new)
        if error < tol
            break
        end
    end
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
    patch = PEPSKit.@autoopt @tensor t[DCLa; DCRa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa2 DNLb; χ2] * env.edges[1,1,1][χ2 DNRa2 DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa2 DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa2 DSRb; χ7] * env.edges[3,1,1][χ7 DSLa2 DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa2 DWb; χ10] * 
    AL_above[DpL; DNLa1 DCLa DSLa1 DWa1] * AR_above[DpR; DNRa1 DEa1 DSRa1 DCRa] * 
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
    patch = PEPSKit.@autoopt @tensor t[DCLa DCLOa; DCRa DCROa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa2 DNLb; χ2] * env.edges[1,1,1][χ2 DNRa2 DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
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
    A::AbstractTensorMap{E,S,1,4},
    Ws::Vector{<:AbstractTensorMap{E,S,1,1}},
    trunc_alg::IntermediateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A, Ws)
    network = InfiniteSquareNetwork(InfinitePEPS(A), InfinitePEPS(A_trunc))
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
    return contract_23_patch(env)
end

function truncation_environment(
    A,
    A_trunc,
    trunc_alg::ApproximateEnvTruncation
)
    t = contract_23_patch()
end