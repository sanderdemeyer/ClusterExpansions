function cost_function_triple(peps, env2, env3, O)
    network = InfiniteSquareNetwork(peps, O)
    return network_value(network, env3) / norm(peps, env2)
end

function fixedpoint_triple(
    operator::InfinitePEPO,
    peps₀::InfinitePEPS,
    env2₀::CTMRGEnv,
    env3₀::CTMRGEnv,
    alg::PEPSOptimize;
    (finalize!)=OptimKit._finalize!,
)
    retract = peps_retract

    # check realness compatibility
    @assert scalartype(env2₀) == scalartype(env3₀) "Scalartype of double and triple environments should be equal"
    if scalartype(env2₀) <: Real && iterscheme(alg.gradient_alg) == :fixed
        env2₀ = complex(env2₀)
        env3₀ = complex(env3₀)
        @warn "the provided real environment was converted to a complex environment since \
        :fixed mode generally produces complex gauges; use :diffgauge mode instead to work \
        with purely real environments"
    end

    # initialize info collection vectors
    T = promote_type(real(scalartype(peps₀)), real(scalartype(env2₀)))
    truncation_errors = Vector{T}()
    condition_numbers = Vector{T}()
    gradnorms_unitcell = Vector{Matrix{T}}()
    times = Vector{Float64}()

    # optimize operator cost function
    (peps_final, env2_final, env3_final), cost, ∂cost, numfg, convergence_history = optimize(
        (peps₀, env2₀, env3₀), alg.optimizer; retract, inner=real_inner, finalize!
    ) do (peps, env2, env3)
        start_time = time_ns()
        E, gs = withgradient(peps) do ψ
            env2′, env3′, info = hook_pullback(
                leading_boundary,
                env2,
                env3,
                ψ,
                alg.boundary_alg;
                alg_rrule=alg.gradient_alg,
            )
            ignore_derivatives() do
                alg.reuse_env && (update!(env2, env2′) && update!(env3, env3′))
                push!(truncation_errors, info.truncation_error)
                push!(condition_numbers, info.condition_number)
            end
            return cost_function_triple(ψ, env2′, env3′, operator)
        end
        g = only(gs)  # `withgradient` returns tuple of gradients `gs`
        push!(gradnorms_unitcell, norm.(g.A))
        push!(times, (time_ns() - start_time) * 1e-9)
        return E, g
    end

    info = (
        last_gradient=∂cost,
        fg_evaluations=numfg,
        costs=convergence_history[:, 1],
        gradnorms=convergence_history[:, 2],
        truncation_errors,
        condition_numbers,
        gradnorms_unitcell,
        times,
    )
    return peps_final, env2_final, env3_final, cost, info
end

# Update PEPS unit cell in non-mutating way
# Note: Both x and η are InfinitePEPS during optimization
function peps_retract(x, η, α)
    peps = deepcopy(x[1])
    peps.A .+= η.A .* α
    env2 = deepcopy(x[2])
    env3 = deepcopy(x[3])
    return (peps, env2, env3), η
end

# Take real valued part of dot product
real_inner(_, η₁, η₂) = real(dot(η₁, η₂))
