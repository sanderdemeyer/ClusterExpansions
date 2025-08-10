abstract type VOPEPO end

struct VOPEPO_CTMRG <: VOPEPO
    ctm_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    miniter::Int
    maxiter::Union{Int,Function}
    α::Number
    check_fidelity::Bool
    verbosity::Int
    ctm_tol::Float64
end

struct VOPEPO_VUMPS <: VOPEPO
    vumps_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    miniter::Int
    maxiter::Union{Int,Function}
    α::Number
    check_fidelity::Bool
    verbosity::Int
    ctm_tol::Float64
end

function VOPEPO_CTMRG(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, miniter::Int, maxiter::T;
    α = 0.5, check_fidelity::Bool = false, verbosity::Int = 0, ctm_tol::Float64 = 1e-7) where {T <: Function}
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, miniter, maxiter, α, check_fidelity, verbosity, ctm_tol)
end

function VOPEPO_CTMRG(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, miniter::Int, maxiter::Int = 2;
    α = 0.5, check_fidelity::Bool = false, verbosity::Int = 0, ctm_tol::Float64 = 1e-7)
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, miniter, i -> maxiter, α, check_fidelity, verbosity, ctm_tol)
end

function VOPEPO_VUMPS(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, miniter::Int, maxiter::T;
    α = 0.5, check_fidelity::Bool = false, verbosity::Int = 0, ctm_tol::Float64 = 1e-7) where {T <: Function}
    return VOPEPO_VUMPS(ctm_alg, envspace, truncspace, miniter, maxiter, α, check_fidelity, verbosity, ctm_tol)
end

function VOPEPO_VUMPS(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, miniter::Int, maxiter::Int = 2;
    α = 0.5, check_fidelity::Bool = false, verbosity::Int = 0, ctm_tol::Float64 = 1e-7)
    return VOPEPO_VUMPS(ctm_alg, envspace, truncspace, miniter, i -> maxiter, α, check_fidelity, verbosity, ctm_tol)
end

function _stack_pepos(pepos)
    return InfinitePEPO(cat(pepos...; dims = 3))
end

function fidelity_vomps(A::Tuple{T,T}, B::T, envspace, ctm_alg) where {E,S,T <: AbstractTensorMap{E,S,2,4}}
    A₁, A₂ = A
    network_B = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    network_A = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(A₂), PEPSKit._dag(A₁))))
    network_overlap = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))

    env_A, = leading_boundary(CTMRGEnv(network_A, envspace), network_A, ctm_alg)
    env_B, = leading_boundary(CTMRGEnv(network_B, envspace), network_B, ctm_alg)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)

    return abs(network_value(network_overlap, env_overlap) / sqrt(network_value(network_A, env_A) * network_value(network_B, env_B)))
end

function get_vomps_A(B, env₀, trunc_alg::VOPEPO_CTMRG)
    network = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    # env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    env, _, η = leading_boundary(CTMRGEnv(network, domain(env₀.corners[1,1,1])[1]), network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor A[DN1 DE1 DS1 DW1; DN2 DE2 DS2 DW2] :=
    env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2; χ2] * 
    env.corners[2,1,1][χ2; χ3] * env.edges[2,1,1][χ3 DE1 DE2; χ4] * 
    env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2; χ6] * 
    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2; χ8]
    return A, env, η
end

function get_vomps_b(A::Tuple{T,T}, B, env₀, trunc_alg::VOPEPO_CTMRG) where {E,S,T <: AbstractTensorMap{E,S,2,4}}
    A₁, A₂ = A
    network = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))
    # env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    env, _, η = leading_boundary(CTMRGEnv(network, domain(env₀.corners[1,1,1])[1]), network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN3 DE3 DS3 DW3] := A₁[Dp Dp1; DN1 DE1 DS1 DW1] * A₂[Dp2 Dp; DN2 DE2 DS2 DW2] * 
    env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2 DN3; χ2] * 
    env.corners[2,1,1][χ2; χ3] * env.edges[2,1,1][χ3 DE1 DE2 DE3; χ4] * 
    env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2 DS3; χ6] * 
    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2 DW3; χ8]
    return b, env, η
end

function get_vomps_A(B, env₀, trunc_alg::VOPEPO_VUMPS)
    network = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor A[DN1 DE1 DS1 DW1; DN2 DE2 DS2 DW2] :=
    env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2; χ2] * 
    env.corners[2,1,1][χ2; χ3] * env.edges[2,1,1][χ3 DE1 DE2; χ4] * 
    env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2; χ6] * 
    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2; χ8]
    return A, env, η
end

function get_vomps_b(A::Tuple{T,T}, B, env₀, trunc_alg::VOPEPO_VUMPS) where {E,S,T <: AbstractTensorMap{E,S,2,4}}
    A₁, A₂ = A
    network = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN3 DE3 DS3 DW3] := A₁[Dp Dp1; DN1 DE1 DS1 DW1] * A₂[Dp2 Dp; DN2 DE2 DS2 DW2] * 
    env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2 DN3; χ2] * 
    env.corners[2,1,1][χ2; χ3] * env.edges[2,1,1][χ3 DE1 DE2 DE3; χ4] * 
    env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2 DS3; χ6] * 
    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2 DW3; χ8]
    return b, env, η
end

function get_vomps_b(A::AbstractTensorMap{E,S,2,4}, B, env₀, trunc_alg::VOPEPO) where {E,S}
    @warn "Using strange version of get_vomps_b"
    network = InfiniteSquareNetwork(_stack_pepos((A, PEPSKit._dag(B))))
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN2 DE2 DS2 DW2] := A[Dp Dp1; DN1 DE1 DS1 DW1] * 
    env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2; χ2] * 
    env.corners[2,1,1][χ2; χ3] * env.edges[2,1,1][χ3 DE1 DE2; χ4] * 
    env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2; χ6] * 
    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2; χ8]
    return b, env, η
end

function apply_vomps_A(A, x::AbstractTensorMap, ::Val{false})
    Ax = ncon([A, x], [vcat(3:6, -3:-1:-6), vcat([-2, -1], 3:6)])
    return permute(Ax, ((1,2),(3,4,5,6)))
end

function apply_vomps_A(A, Ax::AbstractTensorMap, ::Val{true})
    x = ncon([A, Ax], [vcat(-3:-1:-6, 1:4), vcat([-2, -1], 1:4)], [true, false])
    return permute(x, ((1,2),(3,4,5,6)))
end

function initialize_vomps_environments(A₁::T, A₂::T, B::T, trunc_alg) where {T<:AbstractTensorMap}
    network = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    env_double, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    network = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))
    env_triple, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    return env_double, env_triple
end

function initialize_vomps_environments(A₁_space, A₂_space, trunc_alg)
    Ds_double = trunc_alg.truncspace ⊗ trunc_alg.truncspace'
    Ds_triple = A₁_space ⊗ A₂_space ⊗ trunc_alg.truncspace'
    env_double = CTMRGEnv(Ds_double, Ds_double, trunc_alg.envspace)
    env_triple = CTMRGEnv(Ds_triple, Ds_triple, trunc_alg.envspace)
    return env_double, env_triple
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    env_double::CTMRGEnv,
    env_triple::CTMRGEnv,
    trunc_alg::T;
    envspace_fidelity = trunc_alg.envspace,
    ctm_alg_fidelity = trunc_alg.ctm_alg,
    iter = 1
) where {E,S,T<:VOPEPO}
    # if dim(domain(A[1])[1]) == dim(trunc_alg.truncspace)
    #     B = copy(A[1])
    # else
    #     pspace = codomain(A[1])[1]
    #     vspace = trunc_alg.truncspace
    #     B = randn(scalartype(A[1]), pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
    # end
    B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
    if domain(B)[1] == trunc_alg.truncspace
        B = randn(scalartype(B), codomain(B), domain(B))
    else
        B = randn(scalartype(B), codomain(B), trunc_alg.truncspace ⊗ trunc_alg.truncspace ⊗ trunc_alg.truncspace' ⊗ trunc_alg.truncspace')
    end
    ϵ, η_a, η_b = Inf, Inf, Inf
    ϵs = [ϵ]
    Bs = [B]
    for i = 1:trunc_alg.maxiter(iter)
        if trunc_alg.verbosity > 2
            @info "Started with iteration $i. ηs are $η_a and $η_b"
        end
        vomps_A, env_double, η_a = get_vomps_A(B, env_double, trunc_alg)
        vomps_b, env_triple, η_b = get_vomps_b(A, B, env_triple, trunc_alg)
        norm_b = norm(vomps_b)
        vomps_A /= norm_b
        vomps_b /= norm_b
        apply_A = (x, val) -> apply_vomps_A(vomps_A, x, val)
        Bnew, info = lssolve(apply_A, vomps_b, LSMR(verbosity = 0, maxiter = 2000, tol = 1e-16))
        if trunc_alg.verbosity > 0 && info.converged == 0
            @warn "lssolve did not converged to the desired tolerance. residual is $(info.normres)"
        end
        grad = norm(apply_A(B, Val(false)) - vomps_b)
        Bcheck = apply_A(Bnew, Val(false))
        if trunc_alg.check_fidelity
            ϵ = fidelity_vomps(A, Bnew, envspace_fidelity, ctm_alg_fidelity)
            push!(ϵs, ϵ)
            if trunc_alg.verbosity > 2
                @info "Fidelity is $ϵ"
            end
        end
        B = (1-trunc_alg.α)*B + trunc_alg.α*Bnew
        B /= norm(B)
        push!(Bs, B)
        if i >= trunc_alg.miniter && η_a < trunc_alg.ctm_tol && η_b < trunc_alg.ctm_tol && info.converged == 1
            if trunc_alg.verbosity > 1
                @info "Converged after $i iterations with ηs $η_a and $η_b"
            end
            return B, env_double, env_triple, ϵs, Bs
        end
    end
    if trunc_alg.verbosity > 0
        @warn "Not converged after $(trunc_alg.maxiter(iter)) iterations with ηs $η_a and $η_b"
    end
    return B, env_double, env_triple, ϵs, Bs
end

function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    trunc_alg::T
) where {E,S,T<:VOPEPO}
    env_double, env_triple = initialize_vomps_environments(domain(A[1])[1], domain(A[2])[1], trunc_alg)
    return approximate_state(A, env_double, env_triple, trunc_alg)
end