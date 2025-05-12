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

function ExactEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme;
    check_fidelity::Bool = false, verbosity::Int = 0)
    ExactEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, verbosity)
end

function ApproximateEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme;
    check_fidelity::Bool = false, maxiter::Int = 50, tol::Float64 = 1e-10, verbosity::Int = 0)
    ApproximateEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, maxiter, tol, verbosity)
end

function IntermediateEnvTruncation(ctm_alg, envspace::ElementarySpace, trscheme::TruncationScheme;
    check_fidelity::Bool = false, maxiter::Int = 50, tol::Float64 = 1e-10, verbosity::Int = 0)
    IntermediateEnvTruncation(ctm_alg, envspace, trscheme, check_fidelity, maxiter, tol, verbosity)
end

function approximate_state(
    A::Union{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}},
    trunc_alg::EnvTruncation;
    envspace_fidelity = trunc_alg.envspace,
    ctm_alg_fidelity = trunc_alg.ctm_alg
) where {E,S<:ElementarySpace}
    Ws = find_isometry(A, trunc_alg)
    A_trunc = apply_isometry(A, Ws)

    trunc_alg.check_fidelity || return A_trunc, nothing
    overlap = fidelity(A, A_trunc, ctm_alg_fidelity, envspace_fidelity)
    if trunc_alg.verbosity > 0
        @info "Fidelity of approximation is $overlap"
    end
    return A_trunc, overlap
end

function find_isometry(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::ExactEnvTruncation
)
    t = truncation_environment(A, trunc_alg)
    # truncspace = get_trunc_space(A)
    Ws = update_isometry(t, trunc_alg.trscheme)#, truncspace
    return Ws
end

function find_isometry(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace},
    trunc_alg::Union{ApproximateEnvTruncation, IntermediateEnvTruncation}
) 
    T = scalartype(A)
    space = ℂ^(trunc_alg.trscheme.dim)
    Ws = get_initial_isometry(T, get_trunc_space(A), space, randn)
    error = Inf
    for i = 1:trunc_alg.maxiter
        t = truncation_environment(A, Ws, trunc_alg)
        Ws_new = update_isometry(t, trunc_alg.trscheme)
        _, Σ_new, _ = tsvd(Ws_new[1])
        _, Σ, _ = tsvd(Ws[1])
        error = (norm(Σ_new-Σ)/norm(Σ))
        Ws = copy(Ws_new)
        if trunc_alg.verbosity > 1
            @info "Step $i: error = $error"
        end
        if error < trunc_alg.tol
            @info "Converged after $i iterations: error = $error"
            break
        end
        if i == trunc_alg.maxiter && trunc_alg.verbosity > 0
            @warn "Not converged after $i iterations: error = $error"
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

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,2,4},
    AR_above::AbstractTensorMap{E,S,2,4},
    AL_below::AbstractTensorMap{E,S,2,4},
    AR_below::AbstractTensorMap{E,S,2,4},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DCRa; DCLa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa DNLb; χ2] * env.edges[1,1,1][χ2 DNRa DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa DSRb; χ7] * env.edges[3,1,1][χ7 DSLa DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa DWb; χ10] * 
    AL_above[DpbL DptL; DNLa DCLa DSLa DWa] * AR_above[DpbR DptR; DNRa DEa DSRa DCRa] * 
    PEPSKit.dagger(AL_below)[DptL DpbL; DNLb DCb DSLb DWb] * PEPSKit.dagger(AR_below)[DptR DpbR; DNRb DEb DSRb DCb]
    return patch
end

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,2,5},
    AR_above::AbstractTensorMap{E,S,2,5},
    AL_below::AbstractTensorMap{E,S,2,4},
    AR_below::AbstractTensorMap{E,S,2,4},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DCRa DCROa; DCLa DCLOa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa DNLb; χ2] * env.edges[1,1,1][χ2 DNRa DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa DSRb; χ7] * env.edges[3,1,1][χ7 DSLa DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa DWb; χ10] * 
    AL_above[DpbL DptL; DNLa DCLa DSLa DWa DCLOa] * AR_above[DpbR DptR; DNRa DEa DSRa DCRa DCROa] * 
    PEPSKit.dagger(AL_below)[DptL DpbL; DNLb DCb DSLb DWb] * PEPSKit.dagger(AR_below)[DptR DpbR; DNRb DEb DSRb DCb]
    return patch
end

function contract_34_patch(
    AL_above::AbstractTensorMap{E,S,2,5},
    AR_above::AbstractTensorMap{E,S,2,5},
    AL_below::AbstractTensorMap{E,S,2,5},
    AR_below::AbstractTensorMap{E,S,2,5},
    env::CTMRGEnv{C,T}
) where {C,E,S,T<:AbstractTensorMap{E,S,3,1}}
    patch = PEPSKit.@autoopt @tensor t[DCRa DCROa; DCLa DCLOa] := env.corners[1,1,1][χ10; χ1] * env.edges[1,1,1][χ1 DNLa2 DNLb; χ2] * env.edges[1,1,1][χ2 DNRa2 DNRb; χ3] * env.corners[2,1,1][χ3; χ4] * 
    env.edges[2,1,1][χ4 DEa2 DEb; χ5] * env.corners[3,1,1][χ5; χ6] * env.edges[3,1,1][χ6 DSRa2 DSRb; χ7] * env.edges[3,1,1][χ7 DSLa2 DSLb; χ8] * 
    env.corners[4,1,1][χ8; χ9] * env.edges[4,1,1][χ9 DWa2 DWb; χ10] * 
    AL_above[DpbL DptL; DNLa1 DCLa  DSLa1 DWa1 DCLOa] * AR_above[DpbR DptR; DNRa1 DEa1 DSRa1 DCRa DCROa] * 
    PEPSKit.dagger(AL_below)[DptL DpbL; DNLb DCb DSLb DWb DCOb] * PEPSKit.dagger(AR_below)[DptR DpbR; DNRb DEb DSRb DCb DCOb]
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
    A::AbstractTensorMap{E,S,2,4},
    trunc_alg::ExactEnvTruncation
) where {E,S}
    A_stack = fill(A, 1, 1, 2)
    A_stack[1, 1, 2] = dagger(A)
    AAdag = InfinitePEPO(A_stack)

    network = InfiniteSquareNetwork(AAdag)
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    return contract_23_patch(env)
end

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::ExactEnvTruncation
) where {E,S}
    O_stack = fill(A[2], 1, 1, 2)
    O_stack[1, 1, 2] = dagger(A[2])
    OOdag = InfinitePEPO(O_stack)

    network = InfiniteSquareNetwork(InfinitePEPS(A[1]), OOdag)
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    return contract_23_patch(env)
end

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::ExactEnvTruncation
) where {E,S}
    O_stack = fill(A[1], 1, 1, 4)
    O_stack[1, 1, 2] = copy(A[2])
    O_stack[1, 1, 3] = dagger(A[2])
    O_stack[1, 1, 4] = dagger(A[1])
    OOdag = InfinitePEPO(O_stack)

    network = InfiniteSquareNetwork(OOdag)
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
    A::AbstractTensorMap{E,S,2,4},
    Ws::Vector{<:AbstractTensorMap{E,S,1,1}},
    trunc_alg::IntermediateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A, Ws)
    A_stack = fill(A, 1, 1, 2)
    A_stack[1, 1, 2] = dagger(A_trunc)
    AAdag = InfinitePEPO(A_stack)

    network = InfiniteSquareNetwork(AAdag)
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
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    Ws::Vector{<:AbstractTensorMap{E,S,2,1}},
    trunc_alg::IntermediateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A..., Ws)
    A_stack = fill(A[1], 1, 1, 3)
    A_stack[1, 1, 2] = copy(A[2])
    A_stack[1, 1, 3] = dagger(A_trunc)
    AAdag = InfinitePEPO(A_stack)

    network = InfiniteSquareNetwork(AAdag)
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
    A::AbstractTensorMap{E,S,2,4},
    Ws::Vector{<:AbstractTensorMap{E,S,1,1}},
    trunc_alg::ApproximateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A, Ws)
    A_stack = fill(A_trunc, 1, 1, 2)
    A_stack[1, 1, 2] = dagger(A_trunc)
    AAdag = InfinitePEPO(A_stack)

    network = InfiniteSquareNetwork(AAdag)
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

function truncation_environment(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    Ws::Vector{<:AbstractTensorMap{E,S,2,1}},
    trunc_alg::ApproximateEnvTruncation
) where {E,S}
    A_trunc = apply_isometry(A..., Ws)
    A_stack = fill(A_trunc, 1, 1, 2)
    A_stack[1, 1, 2] = dagger(A_trunc)
    AAdag = InfinitePEPO(A_stack)

    network = InfiniteSquareNetwork(AAdag)
    env, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    AL_below = copy(A_trunc)
    AR_below = copy(A_trunc)
    AL_above = apply_isometry(A..., Ws, [2])
    AR_above = apply_isometry(A..., Ws, [4])
    return contract_34_patch(AL_above, AR_above, AL_below, AR_below,env)
end

# Calculate a PEPO-PEPS exactly.
function apply_PEPO_exact(
    ψ::Union{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    O::AbstractTensorMap{E,S,2,4},
) where {E,S}
    T = scalartype(ψ)
    # Not sure whether this works for fermions
    Ws = [dir > 2 ? isometry(T, domain(ψ)[dir] ⊗ domain(O)[dir], fuse(domain(ψ)[dir], domain(O)[dir])') : isometry(T, domain(ψ)[dir] ⊗ domain(O)[dir], fuse(domain(ψ)[dir], domain(O)[dir])) for dir = 1:4]
    return apply_isometry(ψ, O, Ws)
end
