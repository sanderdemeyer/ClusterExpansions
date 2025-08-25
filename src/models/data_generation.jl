function data_generation_SF_CE(time_alg, trunc_alg, χenv; V = 0.0, name = "SF_model_V_$(V).jld2", saving = false)
    ce_alg = spinless_fermion_operators(1.0, V, 0.0; symmetry = nothing, T = Complex{BigFloat})

    # Define observables
    vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
    observables = PEPO_observables([FermionOperators.f_num(), FermionOperators.f_hop(), :spectrum], vumps_alg)
    obs_function = O -> ClusterExpansions.calculate_observables(O, χenv, observables)
    
    βs, expvals, As = time_evolve(ce_alg, time_alg, trunc_alg, obs_function)

    # Extract the expectation values
    ns = [e[1] for e in expvals]
    hops = [e[2] for e in expvals]
    ξs = [e[3][1] for e in expvals]
    δs = [e[3][2] for e in expvals]

    if saving
        file = jldopen(name, "w")
        file["βs"] = βs
        file["ns"] = ns
        file["hops"] = hops
        file["ξs"] = ξs
        file["δs"] = δs
        file["As"] = copy(As)
        close(file)
    end
    return βs, ns, hops, ξs, δs, As
end

function data_generation_ising_CE(time_alg, trunc_alg, χenv; g = 0.0, name = "ising_model_g_$(g).jld2", saving = false)
    ce_alg = ising_operators(1.0, g, 0.0; T = Complex{BigFloat}, symmetry = "C4")

    # Define observables
    vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
    observables = PEPO_observables([SpinOperators.σᶻ(), SpinOperators.σˣ(), :spectrum], vumps_alg)
    observable = O -> ClusterExpansions.calculate_observables(O, χenv, observables)

    βs, expvals, As = time_evolve(ce_alg, time_alg, trunc_alg, observable);

    # Extract the expectation values
    mzs = [e[1] for e in expvals]
    mxs = [e[2] for e in expvals]
    ξs = [e[3][1] for e in expvals]
    δs = [e[3][2] for e in expvals]

    if saving
        file = jldopen(name, "w")
        file["βs"] = βs
        file["mzs"] = mzs
        file["mxs"] = mxs
        file["ξs"] = ξs
        file["δs"] = δs
        file["As"] = copy(As)
        close(file)
    end
    return βs, mzs, mxs, ξs, δs, As
end

function spinless_fermion_model_SU(t, V, μ; T = Float64, Nr = 1, Nc = 1)
    @assert μ == 0.0
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    pspace_fused = fuse(pspace, pspace)

    kinetic_operator = FermionOperators.f_hop(T)
    number_operator = FermionOperators.f_num(T)
    number_operator_halffilling = number_operator - id(pspace)/2
    @tensor number_twosite[-1 -2; -3 -4] := number_operator_halffilling[-1; -3] * number_operator_halffilling[-2; -4]
    twosite_op = rmul!(kinetic_operator, -T(t)) + rmul!(number_twosite, T(V))
    # onesite_op = rmul!(number_operator, -T(μ))

    lattice = InfiniteSquare(Nr, Nc)
    pspaces_fused = fill(pspace_fused, Nr, Nc)

    F = isometry(fuse(pspace,pspace), pspace ⊗ pspace')

    @tensor twosite_final[-1 -2; -3 -4] := twosite_op[1 4; 2 5] * twist(F,3)[-1; 1 3] * twist(F,3)[-2; 4 6] * conj(F[-3; 2 3]) * conj(F[-4; 5 6])
    # @tensor onesite_final[-1; -2] := onesite_op[1; 2] * twist(F,3)[-1; 1 3] * conj(F[-2; 2 3])
        # ((idx,) => onesite_final for idx in PEPSKit.vertices(lattice))...,

    return PEPSKit.LocalOperator(
        pspaces_fused,
        (neighbor => twosite_final for neighbor in PEPSKit.nearest_neighbours(lattice))...,
    )
end

function ising_model_SU(J, g, z; T = Float64, Nr = 1, Nc = 1)
    pspace = ℂ^2
    pspace_fused = fuse(pspace, pspace')

    ZZ = rmul!(PEPSKit.σᶻᶻ(T), -J)
    X = rmul!(PEPSKit.σˣ(T), g * -J) + rmul!(PEPSKit.σᶻ(T), z * -J)

    twosite_op = ZZ + (id(pspace) ⊗ X + X ⊗ id(pspace)) / 4

    lattice = InfiniteSquare(Nr, Nc)
    pspaces_fused = fill(pspace_fused, Nr, Nc)

    F = isometry(fuse(pspace,pspace'), pspace ⊗ pspace')

    @tensor twosite_final[-1 -2; -3 -4] := twosite_op[1 4; 2 5] * twist(F,3)[-1; 1 3] * twist(F,3)[-2; 4 6] * conj(F[-3; 2 3]) * conj(F[-4; 5 6])

    return PEPSKit.LocalOperator(
        pspaces_fused,
        (neighbor => twosite_final for neighbor in PEPSKit.nearest_neighbours(lattice))...,
    )
end

function initialize_state(pspace, trivspace)
    state0 = permute(id(pspace ⊗ trivspace ⊗ trivspace), ((1,4),(5,6,2,3))) * (1 / sqrt(dim(pspace)))
    F = isometry(fuse(pspace,pspace), pspace ⊗ pspace')
    @tensor state[-1; -2 -3 -4 -5] := twist(state0,2)[1 2; -2 -3 -4 -5] * F[-1; 1 2]
    return state
end

function initialize_state_SF()
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    trivspace = Vect[fℤ₂](0 => 1)
    return initialize_state(pspace, trivspace)
end

function initialize_state_ising()
    pspace = ℂ^2
    trivspace = ℂ^1
    return initialize_state(pspace, trivspace)
end

function convert_to_pepo_fuser(A, F)
    @tensor pepo[-1 -2; -3 -4 -5 -6] := A[1; -3 -4 -5 -6] * conj(F[1; -1 -2])
end

function data_generation_SF_SU(time_alg, trunc_alg, χenv; V = 0.0, name = "SF_model_V_$(V)_SU.jld2", saving = false)
    t = 1.0
    μ = 0.0

    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    F = isometry(fuse(pspace,pspace), pspace ⊗ pspace')

    (Nr, Nc) = (2, 2)
    H = spinless_fermion_model_SU(t, V, μ; Nr, Nc)

    state = initialize_state_SF()
    wpeps = InfiniteWeightPEPS(InfinitePEPS(state; unitcell = (Nr, Nc)))

    # Define observables
    # vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
    # observables = PEPO_observables([FermionOperators.f_num(), :spectrum], vumps_alg)
    # obs_function = O -> ClusterExpansions.calculate_observables(O[1,1], χenv, observables)
    
    ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 2)
    observables_SU = PEPO_observables([observable_SU(pspace, FermionOperators.f_num(); Nr, Nc), observable_SU(pspace, FermionOperators.f_hop(); Nr, Nc)], ctm_alg)
    obs_function = O -> ClusterExpansions.calculate_observables(O, χenv, observables_SU)

    tol = 0.0
    maxiter = floor(time_alg.Δt / time_alg.dt)
    βs = [time_alg.dt*maxiter*i for i = 1:time_alg.maxiter]
    convert_to_pepo = A -> convert_to_pepo_fuser(A, F)
    alg = SimpleUpdate(time_alg.dt / 2, tol, maxiter, trunc_alg) # divide by two because we are using the purification here

    expvals = []
    As = []
    for _ = 1:time_alg.maxiter
        result = simpleupdate(wpeps, H, alg; bipartite=false)
        wpeps = result[1]

        peps = InfinitePEPS(wpeps)
        A = InfinitePEPO(reshape(convert_to_pepo.(PEPSKit.unitcell(peps)), Nr, Nc, 1))
        # push!(expvals, obs_function(A))
        push!(expvals, obs_function(peps))
        push!(As, copy(A))
    end

    # Extract the expectation values
    ns = [e[1] for e in expvals]
    hops = [e[2] for e in expvals]
    # ξs = [e[2][1] for e in expvals]
    # δs = [e[2][2] for e in expvals]
    
    if saving
        file = jldopen(name, "w")
        file["βs"] = βs
        file["ns"] = ns
        file["hops"] = hops
        # file["ξs"] = ξs
        # file["δs"] = δs
        file["As"] = copy(As)
    end
    return βs, ns, hops #, ξs, δs, As
end

function calculate_observables_SU(ψ::InfinitePEPS, χenv, observables, ctm_alg)
    envspace = _envspace(codomain(ψ[1,1])[1])(χenv)
    env, = leading_boundary(CTMRGEnv(ψ, envspace), ψ, ctm_alg)

    return [expectation_value(ψ, obs, env) for obs = observables]
end

function data_generation_ising_SU(time_alg, trunc_alg, χenv; g = 0.0, name = "ising_model_g_$(g)_SU.jld2", saving = false)
    J = 1.0
    z = 0.0

    pspace = ℂ^2
    F = isometry(fuse(pspace,pspace'), pspace ⊗ pspace')

    (Nr, Nc) = (2, 2)
    H = ising_model_SU(J, g, z; Nr, Nc)

    state = initialize_state_ising()
    wpeps = InfiniteWeightPEPS(InfinitePEPS(state; unitcell = (Nr, Nc)))

    # Define observables
    vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
    ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 2)
    observables_SU = PEPO_observables([observable_SU(pspace, SpinOperators.σᶻ(); Nr, Nc), observable_SU(pspace, SpinOperators.σˣ(); Nr, Nc), :spectrum], [ctm_alg, ctm_alg, vumps_alg])
    obs_function = O -> ClusterExpansions.calculate_observables(O, χenv, observables_SU)
    
    tol = 0.0
    maxiter = floor(time_alg.Δt / time_alg.dt)
    βs = [time_alg.dt*maxiter*i for i = 1:time_alg.maxiter]
    convert_to_pepo = A -> convert_to_pepo_fuser(A, F)
    alg = SimpleUpdate(time_alg.dt / 2, tol, maxiter, trunc_alg) # divide by two because we are using the purification here

    expvals = []
    As = []
    for _ = 1:time_alg.maxiter
        result = simpleupdate(wpeps, H, alg; bipartite=false)
        wpeps = result[1]

        peps = InfinitePEPS(wpeps)
        A = InfinitePEPO(reshape(convert_to_pepo.(PEPSKit.unitcell(peps)), Nr, Nc, 1))
        push!(expvals, obs_function(peps))
        push!(As, copy(A))
    end

    # Extract the expectation values
    mzs = [e[1] for e in expvals]
    mxs = [e[2] for e in expvals]
    ξs = [e[3][1] for e in expvals]
    δs = [e[3][2] for e in expvals]

    if saving
        file = jldopen(name, "w")
        file["βs"] = βs
        file["mzs"] = mzs
        file["mxs"] = mxs
        file["ξs"] = ξs
        file["δs"] = δs
        file["As"] = copy(As)
        close(file)
    end
    return βs, mzs, mxs, ξs, δs, As
end
