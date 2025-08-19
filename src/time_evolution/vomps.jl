abstract type VOPEPO end

struct VOPEPO_CTMRG <: VOPEPO
    ctm_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    ftol::Number
    gradnormtol::Number
    maxiter::Union{Int,Function}
    verbosity::Int
end

struct VOPEPO_VUMPS <: VOPEPO
    vumps_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    ftol::Number
    gradnormtol::Number
    maxiter::Union{Int,Function}
    verbosity::Int
end

function VOPEPO_CTMRG(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::T;
    verbosity::Int = 0) where {T <: Function}
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter, verbosity)
end

function VOPEPO_CTMRG(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::Int = 2;
    verbosity::Int = 0)
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, i -> maxiter, verbosity)
end

function VOPEPO_VUMPS(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::T;
    verbosity::Int = 0) where {T <: Function}
    return VOPEPO_VUMPS(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter, verbosity)
end

function VOPEPO_VUMPS(ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::Int = 2;
    verbosity::Int = 0)
    return VOPEPO_VUMPS(ctm_alg, envspace, truncspace, ftol, gradnormtol, i -> maxiter, verbosity)
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
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    # env, _, η = leading_boundary(CTMRGEnv(network, domain(env₀.corners[1,1,1])[1]), network, trunc_alg.ctm_alg)
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
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    # env, _, η = leading_boundary(CTMRGEnv(network, domain(env₀.corners[1,1,1])[1]), network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN3 DE3 DS3 DW3] := A₁[Dp Dp1; DN1 DE1 DS1 DW1] * A₂[Dp2 Dp; DN2 DE2 DS2 DW2] * 
    env.corners[1,1,1][χ8; χ1] * env.edges[1,1,1][χ1 DN1 DN2 DN3; χ2] * 
    env.corners[2,1,1][χ2; χ3] * env.edges[2,1,1][χ3 DE1 DE2 DE3; χ4] * 
    env.corners[3,1,1][χ4; χ5] * env.edges[3,1,1][χ5 DS1 DS2 DS3; χ6] * 
    env.corners[4,1,1][χ6; χ7] * env.edges[4,1,1][χ7 DW1 DW2 DW3; χ8]
    return b, env, η
end

function get_vomps_A(B, env₀, trunc_alg::VOPEPO_VUMPS)
    vspace = domain(B)[2]
    F = isometry(fuse(vspace ⊗ vspace'), vspace ⊗ vspace')
        
    @tensor BB_mpo[-3 -4; -2 -1] := B[1 2; 3 5 7 9] * conj(B[1 2; 4 6 8 10]) * conj(F[-1; 3 4]) * conj(F[-2; 5 6]) * F[-3; 7 8] * F[-4; 9 10];
    mpo = InfiniteMPO([BB_mpo])
    
    mps = InfiniteMPS([
        randn(
            scalartype(B),
            trunc_alg.envspace * fuse(vspace * vspace'),
            trunc_alg.envspace,
        )])
    mps, env, η = leading_boundary(mps, mpo, trunc_alg.vumps_alg);

    PEPSKit.@autoopt @tensor A[DN1 DE1 DS1 DW1; DN2 DE2 DS2 DW2] := mps.AC[1][DtL DN; DtR] * conj(mps.AC[1][DbL DS; DbR]) * 
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR] * 
    conj(F[DN; DN1 DN2]) * conj(F[DE; DE1 DE2]) * F[DS; DS1 DS2] * F[DW; DW1 DW2]

    return A, env₀, η
end

function get_vomps_b(A::Tuple{T,T}, B, env₀, trunc_alg::VOPEPO_VUMPS) where {E,S,T <: AbstractTensorMap{E,S,2,4}}
    A₁, A₂ = A
    vspace_B = domain(B)[2]
    vspace_A₁ = domain(A₁)[2]
    vspace_A₂ = domain(A₂)[2]
    F = isometry(fuse(vspace_A₁ ⊗ vspace_A₂ ⊗ vspace_B'), vspace_A₁ ⊗ vspace_A₂ ⊗ vspace_B')
        
    @tensor AAB_mpo[-3 -4; -2 -1] := A₁[2 3; 4 7 10 13] * A₂[1 2; 5 8 11 14] * conj(B[1 3; 6 9 12 15]) * conj(F[-1; 4 5 6]) * conj(F[-2; 7 8 9]) * F[-3; 10 11 12] * F[-4; 13 14 15];
    mpo = InfiniteMPO([AAB_mpo])
    
    mps = InfiniteMPS([
        randn(
            scalartype(B),
            trunc_alg.envspace * fuse(vspace_A₁ ⊗ vspace_A₂ ⊗ vspace_B'),
            trunc_alg.envspace,
        )])
    mps, env, η = leading_boundary(mps, mpo, trunc_alg.vumps_alg);

    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN3 DE3 DS3 DW3] := mps.AC[1][DtL DN; DtR] * conj(mps.AC[1][DbL DS; DbR]) * 
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR] * 
    A₁[Dp Dp1; DN1 DE1 DS1 DW1] * A₂[Dp2 Dp; DN2 DE2 DS2 DW2] * 
    conj(F[DN; DN1 DN2 DN3]) * conj(F[DE; DE1 DE2 DE3]) * F[DS; DS1 DS2 DS3] * F[DW; DW1 DW2 DW3]
    return b, env₀, η
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

# Iterative version of VOPEPO - deprecated
function approximate_state_iteratively(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    env_double::CTMRGEnv,
    env_triple::CTMRGEnv,
    trunc_alg::T;
    envspace_fidelity = trunc_alg.verbosity,
    ctm_alg_fidelity = trunc_alg.verbosity,
    iter = 1
) where {E,S,T<:VOPEPO}
    canoc_alg = Canonicalization()
    B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
    # if domain(B)[1] == trunc_alg.truncspace
    #     B = randn(scalartype(B), codomain(B), domain(B))
    # else
    #     B = randn(scalartype(B), codomain(B), trunc_alg.truncspace ⊗ trunc_alg.truncspace ⊗ trunc_alg.truncspace' ⊗ trunc_alg.truncspace')
    # end
    if domain(B)[1] != trunc_alg.truncspace
        println("Other spaces, initializing with random B")
        B = randn(scalartype(B), codomain(B), trunc_alg.truncspace ⊗ trunc_alg.truncspace ⊗ trunc_alg.truncspace' ⊗ trunc_alg.truncspace')
    end
    ϵ, η_a, η_b = Inf, Inf, Inf
    ϵs = [ϵ]
    Bs = [B]
    for i = 1:trunc_alg.maxiter(iter)
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
        if trunc_alg.verbosity > 2
            @info "Iteration $i. ηs are $η_a and $η_b. Norm of gradient is $(grad)"
        end
        if trunc_alg.check_fidelity
            ϵ = fidelity_vomps(A, Bnew, envspace_fidelity, ctm_alg_fidelity)
            push!(ϵs, ϵ)
            if trunc_alg.verbosity > 2
                @info "Fidelity is $ϵ"
            end
        end
        B = (1-trunc_alg.α)*B + trunc_alg.α*Bnew
        B = canonicalize(B, canoc_alg)
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

function vopepo_costfun_base((pepo, env_double_layer, env_triple_layer), A₁, A₂, boundary_alg)
    rrule_alg = EigSolver(;
    solver_alg=KrylovKit.Arnoldi(; maxiter=30, tol=1e-6, eager=true), iterscheme=:diffgauge
    )
    ## use Zygote to compute the gradient automatically
    E, gs = withgradient(pepo) do B

        ## construct the double and triple layer networks
        n_double_layer = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
        n_triple_layer = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))

        ## contract this network
        env_double_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_double_layer,
            n_double_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        ## contract this network
        env_triple_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_triple_layer,
            n_triple_layer,
            boundary_alg;
            alg_rrule=rrule_alg,
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_double_layer, env_double_layer′)
            PEPSKit.update!(env_triple_layer, env_triple_layer′)
        end

        LHS = PEPSKit.@autoopt @tensor B[Dp1 Dp2; DN1 DE1 DS1 DW1] * conj(B[Dp1 Dp2; DN2 DE2 DS2 DW2]) * 
        env_double_layer.corners[1,1,1][χ8; χ1] * env_double_layer.edges[1,1,1][χ1 DN1 DN2; χ2] * 
        env_double_layer.corners[2,1,1][χ2; χ3] * env_double_layer.edges[2,1,1][χ3 DE1 DE2; χ4] * 
        env_double_layer.corners[3,1,1][χ4; χ5] * env_double_layer.edges[3,1,1][χ5 DS1 DS2; χ6] * 
        env_double_layer.corners[4,1,1][χ6; χ7] * env_double_layer.edges[4,1,1][χ7 DW1 DW2; χ8]
    
        RHS = PEPSKit.@autoopt @tensor conj(B[Dp1 Dp3; DN3 DE3 DS3 DW3]) * 
        A₁[Dp2 Dp3; DN1 DE1 DS1 DW1] * A₂[Dp1 Dp2; DN2 DE2 DS2 DW2] * 
        env_triple_layer.corners[1,1,1][χ8; χ1] * env_triple_layer.edges[1,1,1][χ1 DN1 DN2 DN3; χ2] * 
        env_triple_layer.corners[2,1,1][χ2; χ3] * env_triple_layer.edges[2,1,1][χ3 DE1 DE2 DE3; χ4] * 
        env_triple_layer.corners[3,1,1][χ4; χ5] * env_triple_layer.edges[3,1,1][χ5 DS1 DS2 DS3; χ6] * 
        env_triple_layer.corners[4,1,1][χ6; χ7] * env_triple_layer.edges[4,1,1][χ7 DW1 DW2 DW3; χ8]

        return norm(LHS - RHS)
    end
    g = only(gs)
    return E, g
end

function pepo_retract((peps, env_double_layer, env_triple_layer), η, α)
    peps´, ξ = PEPSKit.norm_preserving_retract(peps, η, α)

    env_double_layer´ = deepcopy(env_double_layer)
    env_triple_layer´ = deepcopy(env_triple_layer)
    return (peps´, env_double_layer´, env_triple_layer´), ξ
end
function pepo_transport!(
    ξ,
    (peps, env_double_layer, env_triple_layer),
    η,
    α,
    (peps´, env_double_layer´, env_triple_layer´),
)
    return PEPSKit.norm_preserving_transport!(
        ξ, peps, η, α, peps´
    )
end;

# AD version of VOPEPO
function approximate_state(
    A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},
    env_double::CTMRGEnv,
    env_triple::CTMRGEnv,
    trunc_alg::T;
    iter = 1
) where {E,S,T<:VOPEPO}
    B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
    if domain(B)[1] != codomain(env_double.edges[1,1,1])[2]
        if trunc_alg.verbosity > 0
            @warn "Other spaces, initializing with random environment"
        end
        env_double, env_triple = initialize_vomps_environments(A[1], A[2], B, trunc_alg)
    end

    optimizer_alg = LBFGS(4; maxiter=trunc_alg.maxiter(iter), gradtol=1e-5, verbosity=3)
    vopepo_costfun = tup -> vopepo_costfun_base(tup, A[1], A[2], trunc_alg.ctm_alg)

    (B_final, env_double_final, env_triple_final), f, = optimize(
    vopepo_costfun,
    (B, env_double, env_triple),
    optimizer_alg;
    inner=PEPSKit.real_inner,
    retract=pepo_retract,
    (transport!)=(pepo_transport!),
    hasconverged = (x, f, g, gradnormhistory) -> (f < trunc_alg.ftol) || (length(gradnormhistory) > 1 && abs(gradnormhistory[end] - gradnormhistory[end-1]) < trunc_alg.gradnormtol)
    );
    return B_final, env_double_final, env_triple_final, f
end
