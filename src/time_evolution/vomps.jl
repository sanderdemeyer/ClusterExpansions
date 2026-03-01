abstract type VOPEPO end

struct VOPEPO_CTMRG <: VOPEPO
    ctm_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    ftol::Number
    gradnormtol::Number
    maxiter::Union{Int, Function}
    c₁::Number
    verbosity::Int
end

struct VOPEPO_VUMPS <: VOPEPO
    vumps_alg
    envspace::ElementarySpace
    truncspace::ElementarySpace
    ftol::Number
    gradnormtol::Number
    maxiter::Union{Int, Function}
    c₁::Number
    verbosity::Int
end

function VOPEPO_CTMRG(
        ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::T;
        c₁::Number = 1.0e-5, verbosity::Int = 0
    ) where {T <: Function}
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter, c₁, verbosity)
end

function VOPEPO_CTMRG(
        ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::Int = 2;
        c₁::Number = 1.0e-5, verbosity::Int = 0
    )
    return VOPEPO_CTMRG(ctm_alg, envspace, truncspace, ftol, gradnormtol, i -> maxiter, c₁, verbosity)
end

function VOPEPO_VUMPS(
        ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::T;
        c₁::Number = 1.0e-5, verbosity::Int = 0
    ) where {T <: Function}
    return VOPEPO_VUMPS(ctm_alg, envspace, truncspace, ftol, gradnormtol, maxiter, c₁, verbosity)
end

function VOPEPO_VUMPS(
        ctm_alg, envspace::ElementarySpace, truncspace::ElementarySpace, ftol::Number, gradnormtol::Number, maxiter::Int = 2;
        c₁::Number = 1.0e-5, verbosity::Int = 0
    )
    return VOPEPO_VUMPS(ctm_alg, envspace, truncspace, ftol, gradnormtol, i -> maxiter, c₁, verbosity)
end

function _stack_pepos(pepos)
    return InfinitePEPO(cat(pepos...; dims = 3))
end

function fidelity_vomps(A::Tuple{T, T}, B::T, envspace, ctm_alg) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
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
        env.corners[1, 1, 1][χ8; χ1] * env.edges[1, 1, 1][χ1 DN1 DN2; χ2] *
        env.corners[2, 1, 1][χ2; χ3] * env.edges[2, 1, 1][χ3 DE1 DE2; χ4] *
        env.corners[3, 1, 1][χ4; χ5] * env.edges[3, 1, 1][χ5 DS1 DS2; χ6] *
        env.corners[4, 1, 1][χ6; χ7] * env.edges[4, 1, 1][χ7 DW1 DW2; χ8]
    return A, env, η
end

function get_vomps_b(A::Tuple{T, T}, B, env₀, trunc_alg::VOPEPO_CTMRG) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
    A₁, A₂ = A
    network = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    # env, _, η = leading_boundary(CTMRGEnv(network, domain(env₀.corners[1,1,1])[1]), network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN3 DE3 DS3 DW3] := A₁[Dp Dp1; DN1 DE1 DS1 DW1] * A₂[Dp2 Dp; DN2 DE2 DS2 DW2] *
        env.corners[1, 1, 1][χ8; χ1] * env.edges[1, 1, 1][χ1 DN1 DN2 DN3; χ2] *
        env.corners[2, 1, 1][χ2; χ3] * env.edges[2, 1, 1][χ3 DE1 DE2 DE3; χ4] *
        env.corners[3, 1, 1][χ4; χ5] * env.edges[3, 1, 1][χ5 DS1 DS2 DS3; χ6] *
        env.corners[4, 1, 1][χ6; χ7] * env.edges[4, 1, 1][χ7 DW1 DW2 DW3; χ8]
    return b, env, η
end

function get_vomps_A(B, env₀, trunc_alg::VOPEPO_VUMPS)
    vspace = domain(B)[2]
    F = isometry(fuse(vspace ⊗ vspace'), vspace ⊗ vspace')

    @tensor BB_mpo[-3 -4; -2 -1] := B[1 2; 3 5 7 9] * conj(B[1 2; 4 6 8 10]) * conj(F[-1; 3 4]) * conj(F[-2; 5 6]) * F[-3; 7 8] * F[-4; 9 10]
    mpo = InfiniteMPO([BB_mpo])

    mps = InfiniteMPS(
        [
            randn(
                scalartype(B),
                trunc_alg.envspace * fuse(vspace * vspace'),
                trunc_alg.envspace,
            ),
        ]
    )
    mps, env, η = leading_boundary(mps, mpo, trunc_alg.vumps_alg)

    PEPSKit.@autoopt @tensor A[DN1 DE1 DS1 DW1; DN2 DE2 DS2 DW2] := mps.AC[1][DtL DN; DtR] * conj(mps.AC[1][DbL DS; DbR]) *
        env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR] *
        conj(F[DN; DN1 DN2]) * conj(F[DE; DE1 DE2]) * F[DS; DS1 DS2] * F[DW; DW1 DW2]

    return A, env₀, η
end

function get_vomps_b(A::Tuple{T, T}, B, env₀, trunc_alg::VOPEPO_VUMPS) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
    A₁, A₂ = A
    vspace_B = domain(B)[2]
    vspace_A₁ = domain(A₁)[2]
    vspace_A₂ = domain(A₂)[2]
    F = isometry(fuse(vspace_A₁ ⊗ vspace_A₂ ⊗ vspace_B'), vspace_A₁ ⊗ vspace_A₂ ⊗ vspace_B')

    @tensor AAB_mpo[-3 -4; -2 -1] := A₁[2 3; 4 7 10 13] * A₂[1 2; 5 8 11 14] * conj(B[1 3; 6 9 12 15]) * conj(F[-1; 4 5 6]) * conj(F[-2; 7 8 9]) * F[-3; 10 11 12] * F[-4; 13 14 15]
    mpo = InfiniteMPO([AAB_mpo])

    mps = InfiniteMPS(
        [
            randn(
                scalartype(B),
                trunc_alg.envspace * fuse(vspace_A₁ ⊗ vspace_A₂ ⊗ vspace_B'),
                trunc_alg.envspace,
            ),
        ]
    )
    mps, env, η = leading_boundary(mps, mpo, trunc_alg.vumps_alg)

    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN3 DE3 DS3 DW3] := mps.AC[1][DtL DN; DtR] * conj(mps.AC[1][DbL DS; DbR]) *
        env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR] *
        A₁[Dp Dp1; DN1 DE1 DS1 DW1] * A₂[Dp2 Dp; DN2 DE2 DS2 DW2] *
        conj(F[DN; DN1 DN2 DN3]) * conj(F[DE; DE1 DE2 DE3]) * F[DS; DS1 DS2 DS3] * F[DW; DW1 DW2 DW3]
    return b, env₀, η
end

function get_vomps_b(A::AbstractTensorMap{E, S, 2, 4}, B, env₀, trunc_alg::VOPEPO) where {E, S}
    @warn "Using strange version of get_vomps_b"
    network = InfiniteSquareNetwork(_stack_pepos((A, PEPSKit._dag(B))))
    env, _, η = leading_boundary(env₀, network, trunc_alg.ctm_alg)
    PEPSKit.@autoopt @tensor b[Dp1 Dp2; DN2 DE2 DS2 DW2] := A[Dp Dp1; DN1 DE1 DS1 DW1] *
        env.corners[1, 1, 1][χ8; χ1] * env.edges[1, 1, 1][χ1 DN1 DN2; χ2] *
        env.corners[2, 1, 1][χ2; χ3] * env.edges[2, 1, 1][χ3 DE1 DE2; χ4] *
        env.corners[3, 1, 1][χ4; χ5] * env.edges[3, 1, 1][χ5 DS1 DS2; χ6] *
        env.corners[4, 1, 1][χ6; χ7] * env.edges[4, 1, 1][χ7 DW1 DW2; χ8]
    return b, env, η
end

function apply_vomps_A(A, x::AbstractTensorMap, ::Val{false})
    Ax = ncon([A, x], [vcat(3:6, -3:-1:-6), vcat([-2, -1], 3:6)])
    return permute(Ax, ((1, 2), (3, 4, 5, 6)))
end

function apply_vomps_A(A, Ax::AbstractTensorMap, ::Val{true})
    x = ncon([A, Ax], [vcat(-3:-1:-6, 1:4), vcat([-2, -1], 1:4)], [true, false])
    return permute(x, ((1, 2), (3, 4, 5, 6)))
end

function initialize_vomps_environments(A₁::T, A₂::T, B::T, trunc_alg) where {T <: AbstractTensorMap}
    network = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    # env_double, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    env_double = CTMRGEnv(network, trunc_alg.envspace)
    network = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))
    # env_triple, = leading_boundary(CTMRGEnv(network, trunc_alg.envspace), network, trunc_alg.ctm_alg)
    env_triple = CTMRGEnv(network, trunc_alg.envspace)
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
        A::Tuple{AbstractTensorMap{E, S, 2, 4}, AbstractTensorMap{E, S, 2, 4}},
        env_double::CTMRGEnv,
        env_triple::CTMRGEnv,
        trunc_alg::T;
        envspace_fidelity = trunc_alg.verbosity,
        ctm_alg_fidelity = trunc_alg.verbosity,
        iter = 1
    ) where {E, S, T <: VOPEPO}
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
    for i in 1:trunc_alg.maxiter(iter)
        vomps_A, env_double, η_a = get_vomps_A(B, env_double, trunc_alg)
        vomps_b, env_triple, η_b = get_vomps_b(A, B, env_triple, trunc_alg)
        norm_b = norm(vomps_b)
        vomps_A /= norm_b
        vomps_b /= norm_b
        apply_A = (x, val) -> apply_vomps_A(vomps_A, x, val)
        Bnew, info = lssolve(apply_A, vomps_b, LSMR(verbosity = 0, maxiter = 2000, tol = 1.0e-16))
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
        B = (1 - trunc_alg.α) * B + trunc_alg.α * Bnew
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
        A::Tuple{AbstractTensorMap{E, S, 2, 4}, AbstractTensorMap{E, S, 2, 4}},
        trunc_alg::T
    ) where {E, S, T <: VOPEPO}
    env_double, env_triple = initialize_vomps_environments(domain(A[1])[1], domain(A[2])[1], trunc_alg)
    return approximate_state(A, env_double, env_triple, trunc_alg)
end

function vopepo_costfun_base((pepo, env_double_layer, env_triple_layer), A₁, A₂, boundary_alg)
    rrule_alg = EigSolver(;
        solver_alg = KrylovKit.Arnoldi(; maxiter = 30, tol = 1.0e-6, eager = true), iterscheme = :diffgauge
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
            alg_rrule = rrule_alg,
        )
        ## contract this network
        env_triple_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_triple_layer,
            n_triple_layer,
            boundary_alg;
            alg_rrule = rrule_alg,
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_double_layer, env_double_layer′)
            PEPSKit.update!(env_triple_layer, env_triple_layer′)
        end

        PEPSKit.@autoopt @tensor LHS[Dpbot; Dptop] := B[Dp Dptop; DN1 DE1 DS1 DW1] * conj(B[Dp Dpbot; DN2 DE2 DS2 DW2]) *
            env_double_layer.corners[1, 1, 1][χ8; χ1] * env_double_layer.edges[1, 1, 1][χ1 DN1 DN2; χ2] *
            env_double_layer.corners[2, 1, 1][χ2; χ3] * env_double_layer.edges[2, 1, 1][χ3 DE1 DE2; χ4] *
            env_double_layer.corners[3, 1, 1][χ4; χ5] * env_double_layer.edges[3, 1, 1][χ5 DS1 DS2; χ6] *
            env_double_layer.corners[4, 1, 1][χ6; χ7] * env_double_layer.edges[4, 1, 1][χ7 DW1 DW2; χ8]

        PEPSKit.@autoopt @tensor RHS[Dpbot; Dptop] := conj(B[Dp1 Dpbot; DN3 DE3 DS3 DW3]) *
            A₁[Dp2 Dptop; DN1 DE1 DS1 DW1] * A₂[Dp1 Dp2; DN2 DE2 DS2 DW2] *
            env_triple_layer.corners[1, 1, 1][χ8; χ1] * env_triple_layer.edges[1, 1, 1][χ1 DN1 DN2 DN3; χ2] *
            env_triple_layer.corners[2, 1, 1][χ2; χ3] * env_triple_layer.edges[2, 1, 1][χ3 DE1 DE2 DE3; χ4] *
            env_triple_layer.corners[3, 1, 1][χ4; χ5] * env_triple_layer.edges[3, 1, 1][χ5 DS1 DS2 DS3; χ6] *
            env_triple_layer.corners[4, 1, 1][χ6; χ7] * env_triple_layer.edges[4, 1, 1][χ7 DW1 DW2 DW3; χ8]
        return norm(LHS - RHS - RHS')
    end
    g = only(gs)
    return E, g
end

function vopepo_costfun_traced_base((pepo, env_double_layer, env_triple_layer), A₁, A₂, boundary_alg)
    rrule_alg = EigSolver(;
        solver_alg = KrylovKit.Arnoldi(; maxiter = 30, tol = 1.0e-6, eager = true), iterscheme = :diffgauge
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
            alg_rrule = rrule_alg,
        )
        ## contract this network
        env_triple_layer′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_triple_layer,
            n_triple_layer,
            boundary_alg;
            alg_rrule = rrule_alg,
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_double_layer, env_double_layer′)
            PEPSKit.update!(env_triple_layer, env_triple_layer′)
        end

        # LHS = PEPSKit.@autoopt @tensor twist(B, 2)[Dp1 Dp2; DN1 DE1 DS1 DW1] * conj(B[Dp1 Dp2; DN2 DE2 DS2 DW2]) *
        # env_double_layer.corners[1,1,1][χ8; χ1] * env_double_layer.edges[1,1,1][χ1 DN1 DN2; χ2] *
        # env_double_layer.corners[2,1,1][χ2; χ3] * env_double_layer.edges[2,1,1][χ3 DE1 DE2; χ4] *
        # env_double_layer.corners[3,1,1][χ4; χ5] * env_double_layer.edges[3,1,1][χ5 DS1 DS2; χ6] *
        # env_double_layer.corners[4,1,1][χ6; χ7] * env_double_layer.edges[4,1,1][χ7 DW1 DW2; χ8]

        # RHS = PEPSKit.@autoopt @tensor conj(B[Dp1 Dp3; DN3 DE3 DS3 DW3]) *
        # twist(A₁, 2)[Dp2 Dp3; DN1 DE1 DS1 DW1] * A₂[Dp1 Dp2; DN2 DE2 DS2 DW2] *
        # env_triple_layer.corners[1,1,1][χ8; χ1] * env_triple_layer.edges[1,1,1][χ1 DN1 DN2 DN3; χ2] *
        # env_triple_layer.corners[2,1,1][χ2; χ3] * env_triple_layer.edges[2,1,1][χ3 DE1 DE2 DE3; χ4] *
        # env_triple_layer.corners[3,1,1][χ4; χ5] * env_triple_layer.edges[3,1,1][χ5 DS1 DS2 DS3; χ6] *
        # env_triple_layer.corners[4,1,1][χ6; χ7] * env_triple_layer.edges[4,1,1][χ7 DW1 DW2 DW3; χ8]

        LHS = network_value(n_double_layer, env_double_layer)
        RHS = network_value(n_triple_layer, env_triple_layer)
        # LHS = PEPSKit.@autoopt @tensor twist(B, 2)[Dp1 Dp2; DN1 DE1 DS1 DW1] * PEPSKit._dag(B)[Dp2 Dp1; DN2 DE2 DS2 DW2] *
        # env_double_layer.corners[1,1,1][χ8; χ1] * env_double_layer.edges[1,1,1][χ1 DN1 DN2; χ2] *
        # env_double_layer.corners[2,1,1][χ2; χ3] * env_double_layer.edges[2,1,1][χ3 DE1 DE2; χ4] *
        # env_double_layer.corners[3,1,1][χ4; χ5] * env_double_layer.edges[3,1,1][χ5 DS1 DS2; χ6] *
        # env_double_layer.corners[4,1,1][χ6; χ7] * env_double_layer.edges[4,1,1][χ7 DW1 DW2; χ8]

        # RHS = PEPSKit.@autoopt @tensor PEPSKit._dag(B)[Dp3 Dp1; DN3 DE3 DS3 DW3] *
        # twist(A₁, 2)[Dp2 Dp3; DN1 DE1 DS1 DW1] * A₂[Dp1 Dp2; DN2 DE2 DS2 DW2] *
        # env_triple_layer.corners[1,1,1][χ8; χ1] * env_triple_layer.edges[1,1,1][χ1 DN1 DN2 DN3; χ2] *
        # env_triple_layer.corners[2,1,1][χ2; χ3] * env_triple_layer.edges[2,1,1][χ3 DE1 DE2 DE3; χ4] *
        # env_triple_layer.corners[3,1,1][χ4; χ5] * env_triple_layer.edges[3,1,1][χ5 DS1 DS2 DS3; χ6] *
        # env_triple_layer.corners[4,1,1][χ6; χ7] * env_triple_layer.edges[4,1,1][χ7 DW1 DW2 DW3; χ8]

        # println("norm of B = $(norm(B)), LHS = $LHS, RHS = $RHS")
        return - real(RHS / sqrt(LHS))
        return real(LHS - RHS - RHS')
        return norm(LHS - RHS) # - RHS')
    end
    g = only(gs)
    return E, g
end

function vomps_monolayer_base((pepo, env_LHS), RHS, obs_function, boundary_alg)
    rrule_alg = EigSolver(;
        solver_alg = KrylovKit.Arnoldi(; maxiter = 30, tol = 1.0e-6, eager = true), iterscheme = :diffgauge
    )
    ## use Zygote to compute the gradient automatically
    E, gs = withgradient(pepo) do B
        ## construct the monolayer networks
        network_LHS = InfiniteSquareNetwork(InfinitePEPO(B))

        ## contract this network
        env_LHS′, info = PEPSKit.hook_pullback(
            leading_boundary,
            env_LHS,
            network_LHS,
            boundary_alg;
            alg_rrule = rrule_alg,
        )
        ## update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env_LHS, env_LHS′)
        end

        # PEPSKit.@autoopt @tensor LHS[Dp1; Dp2] := B[Dp1 Dp2; DN DE DS DW] *
        # env_LHS.corners[1,1,1][χ8; χ1] * env_LHS.edges[1,1,1][χ1 DN; χ2] *
        # env_LHS.corners[2,1,1][χ2; χ3] * env_LHS.edges[2,1,1][χ3 DE; χ4] *
        # env_LHS.corners[3,1,1][χ4; χ5] * env_LHS.edges[3,1,1][χ5 DS; χ6] *
        # env_LHS.corners[4,1,1][χ6; χ7] * env_LHS.edges[4,1,1][χ7 DW; χ8]

        # PEPSKit.@autoopt @tensor LHS[DpL1 DpR1; DpL2 DpR2] := env_LHS.corners[1,1,1][χ10; χ1] * env_LHS.edges[1,1,1][χ1 DNL1; χ2] *
        # env_LHS.edges[1,1,1][χ2 DNR1; χ3] * env_LHS.corners[2,1,1][χ3; χ4] *
        # env_LHS.edges[2,1,1][χ4 DE1; χ5] * env_LHS.corners[3,1,1][χ5; χ6] *
        # env_LHS.edges[3,1,1][χ6 DSR1; χ7] * env_LHS.edges[3,1,1][χ7 DSL1; χ8] *
        # env_LHS.corners[4,1,1][χ8; χ9] * env_LHS.edges[4,1,1][χ9 DW1; χ10] *
        # B[DpL1 DpL2; DNL1 DC1 DSL1 DW1] * B[DpR1 DpR2; DNR1 DE1 DSR1 DC1]

        # (r, c) = (1, 1)
        # LHS = LHS * PEPSKit._contract_corners((r, c), env_LHS) /
        # PEPSKit._contract_vertical_edges((r, c), env_LHS) / PEPSKit._contract_horizontal_edges((r, c), env_LHS)
        # println("Norm of B is $(norm(B)), of RHS = $(norm(RHS)), LHS: $(norm(LHS))")

        LHS = obs_function(B)
        # LHS = network_value_single(B, env_LHS)
        # testing_num = PEPSKit.@autoopt @tensor twist(LHS, (3,4))[DpL1 DpR1; DpL2 DpR2] * M[DpL2 DpR2; DpL1 DpR1]
        # testing_denom = PEPSKit.@autoopt @tensor twist(LHS, (3,4))[DpL1 DpR1; DpL1 DpR1]
        # LHS_hop_operator = testing_num / testing_denom

        # return abs(LHS_hop_operator - RHS_hop_operator)
        return norm(RHS - LHS) / norm(RHS)
    end
    g = only(gs)
    return E, g
end

function pepo_retract((peps, env_monolayer), η, α)
    peps´, ξ = PEPSKit.norm_preserving_retract(peps, η, α)

    env_monolayer′ = deepcopy(env_monolayer)
    return (peps´, env_monolayer′), ξ
end
function pepo_transport!(
        ξ,
        (peps, env_monolayer),
        η,
        α,
        (peps´, env_monolayer′),
    )
    return PEPSKit.norm_preserving_transport!(
        ξ, peps, η, α, peps´
    )
end;

function pepo_retract_triple((peps, env_double_layer, env_triple_layer), η, α)
    peps´, ξ = PEPSKit.norm_preserving_retract(peps, η, α)

    env_double_layer´ = deepcopy(env_double_layer)
    env_triple_layer´ = deepcopy(env_triple_layer)
    return (peps´, env_double_layer´, env_triple_layer´), ξ
end
function pepo_transport_triple!(
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
        A::Tuple{AbstractTensorMap{E, S, 2, 4}, AbstractTensorMap{E, S, 2, 4}},
        env_double::CTMRGEnv,
        env_triple::CTMRGEnv,
        trunc_alg::T;
        iter = 1,
        B::Union{AbstractTensorMap{E, S, 2, 4}, Nothing} = nothing,
    ) where {E, S, T <: VOPEPO}
    if isnothing(B)
        B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
        # B, _ = approximate_state(A, ApproximateEnvTruncation(trunc_alg.ctm_alg, trunc_alg.envspace, truncdim(dim(trunc_alg.truncspace)); maxiter = 4))
    end
    if domain(B)[1] != codomain(env_double.edges[1, 1, 1])[2]
        if trunc_alg.verbosity > 0
            @warn "Other spaces, initializing with random environment"
        end
        env_double, env_triple = initialize_vomps_environments(A[1], A[2], B, trunc_alg)
    end

    # ls_alg = BackTrackingLineSearch(; c₁=trunc_alg.c₁, maxiter=5, maxfg=5);
    ls_alg = HagerZhangLineSearch(; c₁ = trunc_alg.c₁, maxiter = 5, maxfg = 10)
    optimizer_alg = LBFGS(4; maxiter = trunc_alg.maxiter(iter), gradtol = 1.0e-5, verbosity = 3, linesearch = ls_alg) #, scalestep = false)
    vopepo_costfun = tup -> vopepo_costfun_traced_base(tup, A[1], A[2], trunc_alg.ctm_alg)

    (B_final, env_double_final, env_triple_final), f, = optimize(
        vopepo_costfun,
        (B, env_double, env_triple),
        optimizer_alg;
        inner = PEPSKit.real_inner,
        retract = pepo_retract_triple,
        (transport!) = (pepo_transport_triple!),
        hasconverged = (x, fhistory, g, gradnormhistory) -> (length(fhistory) > 1 && abs(fhistory[end] - fhistory[end - 1]) < trunc_alg.ftol) || (length(gradnormhistory) > 1 && abs(gradnormhistory[end] - gradnormhistory[end - 1]) < trunc_alg.gradnormtol)
    )
    return B_final, env_double_final, env_triple_final, f
end

# AD version with observables
function approximate_state_new(
        A::Tuple{AbstractTensorMap{E, S, 2, 4}, AbstractTensorMap{E, S, 2, 4}},
        env_double::CTMRGEnv,
        env_triple::CTMRGEnv,
        trunc_alg::T;
        iter = 1,
        B::Union{AbstractTensorMap{E, S, 2, 4}, Nothing} = nothing,
    ) where {E, S, T <: VOPEPO}
    if isnothing(B)
        B, _ = approximate_state(A, NoEnvTruncation(truncdim(dim(trunc_alg.truncspace))))
        # B, _ = approximate_state(A, ApproximateEnvTruncation(trunc_alg.ctm_alg, trunc_alg.envspace, truncdim(dim(trunc_alg.truncspace)); maxiter = 4))
    end
    if domain(B)[1] != codomain(env_double.edges[1, 1, 1])[2]
        if trunc_alg.verbosity > 0
            @warn "Other spaces, initializing with random environment"
        end
        env_double, env_triple = initialize_vomps_environments(A[1], A[2], B, trunc_alg)
    end
    network_RHS = InfiniteSquareNetwork(_stack_pepos((A[1], A[2])))
    env_RHS, = leading_boundary(CTMRGEnv(network_RHS, trunc_alg.envspace), network_RHS, trunc_alg.ctm_alg)

    network_LHS = InfiniteSquareNetwork(InfinitePEPO(B))
    env_LHS, = leading_boundary(CTMRGEnv(network_LHS, trunc_alg.envspace), network_LHS, trunc_alg.ctm_alg)

    # PEPSKit.@autoopt @tensor RHS[Dp1; Dp2] := A[1][Dp Dp2; DN1 DE1 DS1 DW1] * A[2][Dp1 Dp; DN2 DE2 DS2 DW2] *
    # env_RHS.corners[1,1,1][χ8; χ1] * env_RHS.edges[1,1,1][χ1 DN1 DN2; χ2] *
    # env_RHS.corners[2,1,1][χ2; χ3] * env_RHS.edges[2,1,1][χ3 DE1 DE2; χ4] *
    # env_RHS.corners[3,1,1][χ4; χ5] * env_RHS.edges[3,1,1][χ5 DS1 DS2; χ6] *
    # env_RHS.corners[4,1,1][χ6; χ7] * env_RHS.edges[4,1,1][χ7 DW1 DW2; χ8]

    # PEPSKit.@autoopt @tensor RHS[DpL1 DpR1; DpL2 DpR2] := env_RHS.corners[1,1,1][χ10; χ1] * env_RHS.edges[1,1,1][χ1 DNL1 DNL2; χ2] *
    # env_RHS.edges[1,1,1][χ2 DNR1 DNR2; χ3] * env_RHS.corners[2,1,1][χ3; χ4] *
    # env_RHS.edges[2,1,1][χ4 DE1 DE2; χ5] * env_RHS.corners[3,1,1][χ5; χ6] *
    # env_RHS.edges[3,1,1][χ6 DSR1 DSR2; χ7] * env_RHS.edges[3,1,1][χ7 DSL1 DSL2; χ8] *
    # env_RHS.corners[4,1,1][χ8; χ9] * env_RHS.edges[4,1,1][χ9 DW1 DW2; χ10] *
    # A[1][DpL DpL2; DNL1 DC1 DSL1 DW1] * A[2][DpL1 DpL; DNL2 DC2 DSL2 DW2] *
    # A[1][DpR DpR2; DNR1 DE1 DSR1 DC1] * A[2][DpR1 DpR; DNR2 DE2 DSR2 DC2]

    # (r, c) = (1, 1)
    # RHS = RHS * PEPSKit._contract_corners((r, c), env_RHS) /
    # PEPSKit._contract_vertical_edges((r, c), env_RHS) / PEPSKit._contract_horizontal_edges((r, c), env_RHS)

    # LHS_traced = @tensor twist(LHS, 1)[1; 1]
    # println("LHS is either $(LHS_traced) or $(network_value(network_LHS, env_LHS))")
    # println(done)
    # println("A1 = $(summary(A[1])), A2 = $(summary(A[2])), B = $(summary(B))")
    # RHS = network_value_double(A, env_RHS)

    # M = FermionOperators.f_hop()
    # testing_num = PEPSKit.@autoopt @tensor twist(RHS, (3,4))[DpL1 DpR1; DpL2 DpR2] * M[DpL2 DpR2; DpL1 DpR1]
    # testing_denom = PEPSKit.@autoopt @tensor twist(RHS, (3,4))[DpL1 DpR1; DpL1 DpR1]
    # RHS_hop_operator = testing_num / testing_denom
    # println("Testing exact: $(testing_num) / $(testing_denom), $(testing_num/testing_denom)")

    observables = PEPO_observables([FermionOperators.f_hop()], [trunc_alg.ctm_alg])
    obs_function = O -> calculate_observables(O, dim(trunc_alg.envspace), observables)
    RHS = obs_function(apply_PEPO_exact(A[1], A[2]))

    # ls_alg = BackTrackingLineSearch(; c₁=trunc_alg.c₁, maxiter=50, maxfg=50);
    ls_alg = HagerZhangLineSearch(; c₁ = trunc_alg.c₁, maxiter = 50, maxfg = 50)
    optimizer_alg = LBFGS(64; maxiter = trunc_alg.maxiter(iter), gradtol = 1.0e-5, verbosity = 2, linesearch = ls_alg) #, scalestep = false)
    vomps_monolayer = tup -> vomps_monolayer_base(tup, RHS, obs_function, trunc_alg.ctm_alg)
    (B_final, env_LHS_final), f, = optimize(
        vomps_monolayer,
        (B, env_LHS),
        optimizer_alg;
        inner = PEPSKit.real_inner,
        retract = pepo_retract,
        (transport!) = (pepo_transport!),
        hasconverged = (x, fhistory, g, gradnormhistory) -> (length(fhistory) > 11 && abs(fhistory[end] - fhistory[end - 10]) < trunc_alg.ftol) || (length(gradnormhistory) > 1 && abs(gradnormhistory[end] - gradnormhistory[end - 1]) < trunc_alg.gradnormtol)
    )
    return B_final, env_LHS_final, f
end
