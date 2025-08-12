struct PEPOObservable
    obs::Union{LocalOperator, InfinitePEPO, TensorMap, Function, Symbol}
    env_alg::Union{PEPSKit.CTMRGAlgorithm,VUMPS,Nothing}
end

PEPOObservable(obs::Function) = PEPOObservable(obs, nothing)

function PEPO_observables(obss::Vector, env_algs::Vector)
    return [PEPOObservable(obs, env_alg) for (obs, env_alg) in zip(obss, env_algs)]
end

function PEPO_observables(obss::Vector, env_alg::Union{PEPSKit.CTMRGAlgorithm,VUMPS,Nothing})
    return [PEPOObservable(obs, env_alg) for obs in obss]
end

function localoperator_model(pspace, op::TensorMap{T,S,1,1}; Nr = 1, Nc = 1) where {T,S}
    pspaces = fill(pspace, Nr, Nc)
    lattice = InfiniteSquare(Nr, Nc)
    return PEPSKit.LocalOperator(pspaces, ((idx,) => op for idx in PEPSKit.vertices(lattice))...,)
end

function localoperator_model(pspace, op::TensorMap{T,S,2,2}; Nr = 1, Nc = 1) where {T,S}
    pspaces = fill(pspace, Nr, Nc)
    lattice = InfiniteSquare(Nr, Nc)
    return PEPSKit.LocalOperator(pspaces, (neighbor => op for neighbor in PEPSKit.nearest_neighbours(lattice))...,)
end

function localoperator_model(pspace, op::Tuple{TensorMap{T,S,1,1},TensorMap{T,S,2,2}}; Nr = 1, Nc = 1) where {T,S}
    pspaces = fill(pspace, Nr, Nc)
    lattice = InfiniteSquare(Nr, Nc)
    return PEPSKit.LocalOperator(pspaces, (neighbor => op[2] for neighbor in PEPSKit.nearest_neighbours(lattice))...,
                                            ((idx,) => op[1] for idx in vertices(lattice))...,)
end

function MPSKit.expectation_value(ρ::InfinitePEPO, obs::Union{TensorMap{T,S,1,1},TensorMap{T,S,2,2}}, env::CTMRGEnv) where {T,S}
    pspace = domain(obs)[1]
    H = localoperator_model(pspace, obs)
    return expectation_value(ρ, H, env)
end

function MPSKit.expectation_value(ρ::InfinitePEPO, obs::TensorMap{T,S,1,1}, (mps,env)::Tuple) where {T,S}
    O = ρ[1,1]
    E_num = PEPSKit.@autoopt @tensor twist(O,2)[d1 d2; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) * obs[d2; d1] * 
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(O,2)[d d; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) *
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    return E_num / E_denom
end

function MPSKit.expectation_value(ρ::InfinitePEPO, obs::TensorMap{T,S,2,2}, (mps,env)::Tuple) where {T,S}
    O = ρ[1,1]
    E_num = PEPSKit.@autoopt @tensor twist(O,2)[dL1 dL2; DLN Dc DLS DLW] * twist(O,2)[dR1 dR2; DRN DRE DRS Dc] * mps.AC[1][DtL DLN; Dt] * 
    conj(mps.AC[1][DbL DLS; Db]) * mps.AR[2][Dt DRN; DtR] * conj(mps.AR[2][Db DRS; DbR]) * obs[dL2 dR2; dL1 dR1] * 
    env.GLs[1][DbL DLW; DtL] * env.GRs[1][DtR DRE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(O,2)[dL dL; DLN Dc DLS DLW] * twist(O,2)[dR dR; DRN DRE DRS Dc] * mps.AC[1][DtL DLN; Dt] * 
    conj(mps.AC[1][DbL DLS; Db]) * mps.AR[2][Dt DRN; DtR] * conj(mps.AR[2][Db DRS; DbR]) *
    env.GLs[1][DbL DLW; DtL] * env.GRs[1][DtR DRE; DbR]
    return E_num / E_denom
end

function MPSKit.expectation_value(::InfinitePEPO, symb::Symbol, (mps,env)::Tuple)
    if symb == :spectrum
        ϵ, δ, = marek_gap(mps; num_vals = 20)
        return 1 / ϵ, δ
    else
        @warn "Observable $(symb) not defined. This will be set to zero"
        return 0
    end
end

function MPSKit.expectation_value(ψ::InfinitePEPS, symb::Symbol, env::CTMRGEnv)
    if symb == :spectrum
        ξ_h, ξ_v, λ_h, λ_v = correlation_length(ψ, env; num_vals = 3)
        return ξ_h, log(abs(λ_h[2])/log(λ_h[3]))
    else
        @warn "Observable $(symb) not defined. This will be set to zero"
        return 0
    end
end

function _env_algs(observables::Vector{PEPOObservable})
    return [obs.env_alg isa VUMPS ? :VUMPS : (obs.env_alg isa PEPSKit.CTMRGAlgorithm ? :CTMRG : :nothing) for obs = observables]
end

function get_env_alg(observables::Vector{PEPOObservable}, alg_type)
    algs = [obs.env_alg for obs = observables if obs.env_alg isa alg_type]
    if !all([algs[1] == alg for alg = algs]) 
        @warn "Algorithms are different. Using only the first one"
    end
    if algs == []
        return nothing
    end
    return algs[1]
end

function calculate_observables(O::AbstractTensorMap{E,S,2,4}, χ::Int, observables) where {E,S}
    envspace = _envspace(codomain(O)[1])(χ)
    env_algs = _env_algs(observables)

    vumps_alg = get_env_alg(observables, VUMPS)
    ctm_alg = get_env_alg(observables, PEPSKit.CTMRGAlgorithm)
    ρ = InfinitePEPO(O)
    pf = PEPSKit.trace_out(ρ)
    if :VUMPS ∈ env_algs
        T = InfiniteMPO([pf[1,1]])
        pspace = domain(pf[1,1])[2]
    
        mps = InfiniteMPS([
            randn(
                ComplexF64,
                envspace * pspace,
                envspace,
            )])
        mps, env, _ = leading_boundary(mps, T, vumps_alg)
        vumps_env = (mps, env)
    end
    if :CTMRG ∈ env_algs
        ctmrg_env, = leading_boundary(CTMRGEnv(pf, envspace), pf, ctm_alg)
    end
    return [env_type == :VUMPS ? expectation_value(ρ, obs.obs, vumps_env) : (env_type == :CTMRG ? expectation_value(ρ, obs.obs, ctmrg_env) : obs.obs(ρ)) for (obs,env_type) = zip(observables, env_algs)]
end

# Utility functions for Simple Update

function observable_SU(pspace, obs::AbstractTensorMap{T,S,1,1}; Nr = 1, Nc = 1) where {T,S}
    pspace_fused = fuse(pspace,pspace')
    pspaces_fused = fill(pspace_fused, Nr, Nc)
    lattice = InfiniteSquare(Nr, Nc)
    F = isometry(pspace_fused, pspace ⊗ pspace')
    @tensor obs_final[-1; -2] := obs[1; 2] * twist(F,3)[-1; 1 3] * conj(F[-2; 2 3])

    return PEPSKit.LocalOperator(
        pspaces_fused,
        ((idx,) => obs_final for idx in PEPSKit.vertices(lattice))...,
    )
end

function observable_SU(pspace, obs::AbstractTensorMap{T,S,2,2}; Nr = 1, Nc = 1) where {T,S}
    pspace_fused = fuse(pspace,pspace')
    pspaces_fused = fill(pspace_fused, Nr, Nc)
    lattice = InfiniteSquare(Nr, Nc)
    F = isometry(pspace_fused, pspace ⊗ pspace')
    @tensor obs_final[-1 -2; -3 -4] := obs[1 4; 2 5] * twist(F,3)[-1; 1 3] * twist(F,3)[-2; 4 6] * conj(F[-3; 2 3]) * conj(F[-4; 5 6])

    return PEPSKit.LocalOperator(
        pspaces_fused,
        (neighbor => obs_final for neighbor in PEPSKit.nearest_neighbours(lattice))...,
    )
end

function calculate_observables(ψ::InfinitePEPS, χ::Int, observables)
    envspace = _envspace(codomain(ψ[1,1])[1])(χ)
    env_algs = _env_algs(observables)

    # vumps_alg = get_env_alg(observables, VUMPS)
    ctm_alg = get_env_alg(observables, PEPSKit.CTMRGAlgorithm)

    if :VUMPS ∈ env_algs
        @error "TBA"
    end
    if :CTMRG ∈ env_algs
        ctmrg_env, = leading_boundary(CTMRGEnv(ψ, envspace), ψ, ctm_alg)
    end
    return [env_type == :VUMPS ? expectation_value(ψ, obs.obs, vumps_env) : (env_type == :CTMRG ? expectation_value(ψ, obs.obs, ctmrg_env) : obs.obs(ψ)) for (obs,env_type) = zip(observables, env_algs)]
end
