## Symmetrization, retraction and finalization

function pepo_retract_3layer((peps, env_double_layer, env_triple_layer), η, α)
    (peps´, env_double_layer´), ξ = PEPSKit.peps_retract((peps, env_double_layer), η, α)
    env_triple_layer´ = deepcopy(env_triple_layer)
    return (peps´, env_double_layer´, env_triple_layer´), ξ
end

function pepo_transport_3layer!(
        ξ,
        (peps, env_double_layer, env_triple_layer),
        η,
        α,
        (peps´, env_double_layer´, env_triple_layer´),
    )
    return PEPSKit.peps_transport!(
        ξ, (peps, env_double_layer), η, α, (peps´, env_double_layer´)
    )
end

"""
    pepo_symmetrize_retract_and_finalize!(symm, [retract, finalize!])

Return the `retract` and `finalize!` function for symmetrizing the `peps` and `grad` tensors
during a PEPO fixed point optimization.
"""
function pepo_symmetrize_retract_and_finalize!(
        symm, retract = pepo_retract_3layer, (finalize!) = OptimKit._finalize!
    )

    isnothing(symm) && return retract, finalize!

    function symmetrize_then_finalize!((peps, env2, env3), E, grad, numiter)
        # symmetrize the gradient
        grad_symm = symmetrize!(grad, symm)
        # then finalize
        return finalize!((peps, env2, env3), E, grad_symm, numiter)
    end
    function retract_then_symmetrize((peps, env2, env3), η, α)
        # retract
        (peps´, env2´, env3´), ξ = retract((peps, env2, env3), η, α)
        # symmetrize retracted point and directional derivative
        peps´_symm = symmetrize!(peps´, symm)
        ξ_symm = symmetrize!(ξ, symm)
        return (peps´_symm, env2´, env3´), ξ_symm
    end
    return retract_then_symmetrize, symmetrize_then_finalize!
end

## CTMRG-based optimization

function generate_ctmrg_pepo_cost_function(
        T::InfinitePEPO;
        symmetrization = nothing,
        boundary_alg = SimultaneousCTMRG(),
        gradient_alg = FixedPointGradient(),
    )
    function pepo_costfun((peps, env2, env3))
        ## use Zygote to compute the gradient automatically
        E, gs = withgradient(peps) do ψ
            ## construct the PEPS norm network
            n2 = InfiniteSquareNetwork(ψ)
            ## contract this network
            env2′, info = PEPSKit.hook_pullback(
                leading_boundary,
                env2,
                n2,
                boundary_alg;
                alg_rrule = gradient_alg,
            )
            ## construct the PEPS-PEPO-PEPS overlap network
            n3 = InfiniteSquareNetwork(ψ, T)
            ## contract this network
            env3′, info = PEPSKit.hook_pullback(
                leading_boundary,
                env3,
                n3,
                boundary_alg;
                alg_rrule = gradient_alg,
            )
            ## update the environments for reuse
            PEPSKit.ignore_derivatives() do
                PEPSKit.update!(env2, env2′)
                PEPSKit.update!(env3, env3′)
            end
            ## compute the network values per site
            λ3 = network_value(n3, env3)
            λ2 = network_value(n2, env2)
            ## use this to compute the actual cost function
            return -log(abs(λ3 / λ2))
            # return -log(real(λ3 / λ2))
        end
        g = only(gs)
        symmetrize!(g, symmetrization)
        return E, g
    end

    return pepo_costfun
end

## VUMPS-based optimization

# this is a bit hacky for now...

mutable struct MPSEnv{A, B, G}
    mps::InfiniteMPS{A, B}
    envs::MPSKit.InfiniteEnvironments{G}
end

mps(env::MPSEnv) = env.mps
envs(env::MPSEnv) = env.envs

function PEPSKit.update!(env::MPSEnv{A, B, G}, env´::MPSEnv{A, B, G}) where {A, B, G}
    env.mps = env´.mps
    env.envs = env´.envs
    return env
end

function _to_mpo(n::InfiniteSquareNetwork)
    @assert size(n, 1) == 1 "Only single-row network supported for now" # TODO: relax this    return
    return InfiniteMPO(PeriodicArray(PEPSKit.unitcell(n)[1, :]))
end

function MPSEnv(n::InfiniteSquareNetwork, Venv::ElementarySpace)
    T = _to_mpo(n)
    mps = initialize_mps(T, [Venv])
    envs = MPSKit.InfiniteEnvironments(MPSKit.initialize_environments(mps, T)...)
    return MPSEnv(mps, envs)
end

function MPSKit.leading_boundary(
        env₀::MPSEnv, network::InfiniteSquareNetwork, alg::VUMPS
    )
    T = _to_mpo(network)
    mps´, envs´, ϵ = leading_boundary(mps(env₀), T, alg, envs(env₀))
    λ = expectation_value(mps´, T, envs´)
    return MPSEnv(mps´, envs´), λ, ϵ
end

_drop_bra(O::PEPSKit.PEPSSandwich) = PEPSKit.ket(O)
_drop_bra(O::PEPSKit.PEPOSandwich) = (PEPSKit.ket(O), PEPSKit.pepo(O)...)

function PEPSKit.∂peps(network::InfiniteSquareNetwork, env::MPSEnv)
    @assert size(network, 1) == 1 "Only single-row network supported for now"
    dpepss = map(eachindex(IndexCartesian(), PEPSKit.unitcell(network))) do c
        i, j = Tuple(c)
        AC = mps(env).AC[j]
        O = _drop_bra(network[i, j])
        GL = envs(env).GLs[j]
        GR = envs(env).GRs[j]
        return PEPSKit.∂peps(AC, AC, O, GL, GR)
    end
    return InfinitePEPS(dpepss)
end

# implementation of algorithm in Appendix of https://arxiv.org/abs/2102.03143
function generate_vumps_pepo_cost_function(
        T::InfinitePEPO;
        symmetrization = nothing,
        boundary_alg = VUMPS(),
    )
    function pepo_costfun((peps, env2, env3))
        # contract the networks
        n2 = InfiniteSquareNetwork(peps)
        env2′, λ2, = leading_boundary(env2, n2, boundary_alg)
        n3 = InfiniteSquareNetwork(peps, T)
        env3′, λ3, = leading_boundary(env3, n3, boundary_alg)
        # update the environments for reuse
        PEPSKit.ignore_derivatives() do
            PEPSKit.update!(env2, env2′)
            PEPSKit.update!(env3, env3′)
        end
        # use this to compute the cost function
        f = try
            -log(real(λ3 / λ2))
        catch
            @error "Cost function evaluation failed, returning Inf"
            Inf
        end
        # and get the gradient
        ∂p2 = PEPSKit.∂peps(n2, env2)
        ∂p3 = PEPSKit.∂peps(n3, env3)
        g = - 2 / λ3 * ∂p3 + 2 / λ2 * ∂p2
        # symmetrize the gradient
        symmetrize!(g, symmetrization)
        return f, g
    end

    return pepo_costfun
end

function vumpsify(alg::PEPSKit.CTMRGAlgorithm)
    return VUMPS(;
        tol = alg.tol,
        maxiter = alg.maxiter,
        verbosity = alg.verbosity,
    )
end


## Observables

function compute_magnetization(T::InfinitePEPO, M::PEPSKit.PEPOTensor, peps::InfinitePEPS, env3::CTMRGEnv)
    n3 = InfiniteSquareNetwork(peps, T)
    num = PEPSKit.contract_local_tensor((1, 1, 1), M, n3, env3)
    denom = PEPSKit._contract_site((1, 1), n3, env3)
    m = num / denom
    abs(imag(m)) < 1.0e-12 || @warn "Magnetization has a significant imaginary part: $(m)"
    return real(m)
end

function compute_magnetization(T::InfinitePEPO, M::PEPSKit.PEPOTensor, peps::InfinitePEPS, env3::MPSEnv)
    # just wrapped it because I need the name
    m = compute_expval(T, M, peps, env3)
    abs(imag(m)) < 1.0e-12 || @warn "Magnetization has a significant imaginary part: $(m)"
    return real(m)
end

function compute_expval(T::InfinitePEPO, M::PEPSKit.PEPOTensor, peps::InfinitePEPS, env3::MPSEnv)
    @assert size(peps) == (1, 1) "Only single-site network supported for now"
    AC = mps(env3).AC[1]
    GL = envs(env3).GLs[1]
    GR = envs(env3).GRs[1]
    O_sandwich = (peps[1, 1], peps[1, 1], T[1, 1, 1])
    M_sandwich = (peps[1, 1], peps[1, 1], M)

    num = MPSKit.contract_mpo_expval(AC, GL, M_sandwich, GR, AC)
    denom = MPSKit.contract_mpo_expval(AC, GL, O_sandwich, GR, AC)

    return num / denom
end

## Piracy

# overwrite PEPSKit.initialize_random_c4v_env to make this work directly on InfiniteSquareNetworks
function initialize_random_c4v_env(network::Union{InfiniteSquareNetwork, InfinitePartitionFunction, InfinitePEPS}, Venv::ElementarySpace)
    return initialize_random_c4v_env(randn, scalartype(network), network, Venv)
end
function initialize_random_c4v_env(f, T, state::Union{InfinitePartitionFunction, InfinitePEPS}, Venv::ElementarySpace)
    return initialize_random_c4v_env(f, T, InfiniteSquareNetwork(state), Venv::ElementarySpace)
end
function initialize_random_c4v_env(f, T, network::InfiniteSquareNetwork, Venv::ElementarySpace)
    Vstate = only(PEPSKit._north_edge_physical_spaces(network))
    return initialize_random_c4v_env(f, T, Vstate, Venv)
end
function initialize_random_c4v_env(f, T, Vstate::VectorSpace, Venv::ElementarySpace)
    corner₀ = DiagonalTensorMap(randn(real(T), Venv ← Venv))
    edge₀ = f(T, Venv ⊗ Vstate ← Venv)
    edge₀ = PEPSKit._project_hermitian(edge₀)
    return CTMRGEnv(corner₀, edge₀)
end
