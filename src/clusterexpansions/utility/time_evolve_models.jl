# function pf_observable(O::PEPSKit.PEPOTensor, env0, M, ctm_alg)
#     @tensor Z[-3 -4; -1 -2] := O[1 1; -1 -2 -3 -4]
#     partfunc = InfinitePartitionFunction(Z)
#     @tensor A_M[-3 -4; -1 -2] := O[1 2; -1 -2 -3 -4] * M[2; 1]
#     partfunc_M = InfinitePartitionFunction(A_M)

#     if codomain(env0.edges[1,1,1])[2] != domain(Z)[1]
#         env0 = CTMRGEnv(partfunc, codomain(env0.edges[1,1,1])[1])
#     end        
#     env, = leading_boundary(env0, partfunc, ctm_alg)

#     Z = network_value(partfunc, env);
#     obs = network_value(partfunc_M, env)
#     if abs(imag(obs) / real(obs)) > 1e-5
#         @warn "complex value for observable: $(imag(obs) / real(obs))"
#     end
#     return real(obs / Z), env
# end

# function pf_observable(A::PEPSKit.PEPSTensor, env0, H, ctm_alg)
#     # peps = InfinitePEPS(A)
#     # pspace = codomain(A)[1]
#     # pspaces = fill(pspace, 1, 1)
#     # H = PEPSKit.LocalOperator(pspaces, (CartesianIndex(1, 1),) => M)

#     if codomain(env0.edges[1,1,1])[2] != domain(A)[1]
#         env0 = CTMRGEnv(peps, codomain(env0.edges[1,1,1])[1])
#     end
#     env, = leading_boundary(env0, peps, ctm_alg)
#     obs = expectation_value(peps, H, env)
#     if abs(imag(obs) / real(obs)) > 1e-5
#         @warn "complex value for observable: $(imag(obs) / real(obs))"
#     end
#     return real(obs), env
# end

function localoperator_model(pspace, op; Nr = 1, Nc = 1)
    pspaces = fill(pspace, Nr, Nc)
    lattice = InfiniteSquare(Nr, Nc)
    return PEPSKit.LocalOperator(pspaces, ((idx,) => op for idx in PEPSKit.vertices(lattice))...,)
end

function observable_time_evolve(O::AbstractTensorMap{T,S,2,4}, observable::Union{InfinitePEPO,PEPSKit.LocalOperator}, envspace, ctm_alg; convert_symm = false) where {T,S}
    if convert_symm
        codom_asym = [(i == 2) ? ComplexSpace(dim(codomain(O)[i]))' : ComplexSpace(dim(codomain(O)[i])) for i = 1:2]
        dom_asym = [(i > 2) ? ComplexSpace(dim(domain(O)[i]))' : ComplexSpace(dim(domain(O)[i])) for i = 1:4]
        O_asym = TensorMap(convert(Array, O), prod(codom_asym), prod(dom_asym))
        # O_asym += 1e-5*randn(ComplexF64, prod(codom_asym), prod(dom_asym))
        return expectation_value(InfinitePEPO(O_asym), observable, ComplexSpace(dim(envspace)), ctm_alg)
    end
    return expectation_value(InfinitePEPO(O), observable, envspace, ctm_alg)
end

function observable_time_evolve(O::AbstractTensorMap{T,S,1,4}, observable::PEPSKit.LocalOperator, envspace, ctm_alg; convert_symm = false) where {T,S}
    env, = leading_boundary(CTMRGEnv(InfinitePEPS(O), envspace), InfinitePEPS(O), ctm_alg)
    return expectation_value(InfinitePEPS(O), observable, env)
end

function observable_time_evolve(O::Union{AbstractTensorMap{T,S,1,4},AbstractTensorMap{T,S,2,4}}, observable::F, envspace, ctm_alg; convert_symm = false) where {T,S,F<:Function}
    return observable(O, envspace, ctm_alg)
end

function time_evolve_model(model, param, time_alg, χenv, trscheme, trscheme_parameters; trscheme_kwargs = (), ce_kwargs = (), observables = [σˣ()], convert_symm = false, verbosity_ctm = 0, verbosity_trunc = 0, A0 = nothing, finalize! = nothing)
    ce_alg = model(param...; ce_kwargs...)
    envspace = ce_alg.envspace(χenv)

    ctm_alg = SimultaneousCTMRG(;
        tol=1e-10,
        miniter=4,
        maxiter=250,
        verbosity=verbosity_ctm,
        svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
    )

    trunc_alg = trscheme(trscheme_parameters...; verbosity = verbosity_trunc, trscheme_kwargs...)
    obs_function = O -> [observable_time_evolve(O, obs, envspace, ctm_alg; convert_symm) for obs in observables]
    times, expvals, As = time_evolve(ce_alg, time_alg, trunc_alg, obs_function; A0, finalize!)
    return times, expvals, As
end

function time_scan_model(model, param, times, χenv; T = ComplexF64, observables = [σˣ()], verbosity = 0, ctm_alg = nothing, p = 3)
    ce_alg = model(param...; T, p)
    envspace = ce_alg.envspace(χenv)

    if isnothing(ctm_alg)
        ctm_alg = SimultaneousCTMRG(;
        tol=1e-10,
        miniter=4,
        maxiter=100,
        verbosity=2,
        svd_alg=SVDAdjoint(; fwd_alg=TensorKit.SVD(), rrule_alg=GMRES(; tol=1e-10))
        )
    end
    return time_scan(ce_alg, times,  O -> observable_time_evolve(O, observables, envspace, ctm_alg; convert_symm); verbosity)
end
