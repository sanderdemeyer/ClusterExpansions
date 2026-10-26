function time_evolve_model(model, param, time_alg, χenv, trscheme, trscheme_parameters; trscheme_kwargs = (), ce_kwargs = (), observables = PEPO_observables([], SimultaneousCTMRG(;)), verbosity_trunc = 0, A0 = nothing, finalize! = nothing)
    ce_alg = model(param...; ce_kwargs...)
    obs_function = (O,i) -> calculate_observables(O, χenv, observables)

    trunc_alg = trscheme(trscheme_parameters...; verbosity = verbosity_trunc, trscheme_kwargs...)

    return time_evolve(ce_alg, time_alg, trunc_alg, obs_function; A0, finalize!)
end

function time_scan_model(model, param, times, χenv; ce_kwargs = (), observables = PEPO_observables([], SimultaneousCTMRG(;)), verbosity = 0)
    ce_alg = model(param...; ce_kwargs...)
    obs_function = (O,i) -> calculate_observables(O, χenv, observables)

    return time_scan(ce_alg, times,  obs_function; verbosity)
end
