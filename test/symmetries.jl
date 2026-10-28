using TensorKit
using ClusterExpansions
using PEPSKit
using TensorKitTensors
using Test
using Random

Random.seed!(14071789)

models = [ising_operators, heisenberg_operators, spinless_fermion_operators]
params = [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0), (1.0, 1.0, 0.5)]
symmetry_list = [[Trivial, Z2Irrep], (Trivial, Z2Irrep, U1Irrep, SU2Irrep), (Trivial,)]
operators = [SpinOperators.S_z_S_z, SpinOperators.S_exchange, FermionOperators.f_num]
# Parameters in the truncation scheme
Dcut = 4
trunc_alg = NoEnvTruncation(truncrank(Dcut))
χenv = 20 # Environment bond dimension used in the calculation of expectation values

# Set up time evolution algorithm
β = 0.05
ctm_alg = SimultaneousCTMRG(; maxiter = 1500, verbosity = 1)

ps = [3, 4]

for (i,(model,param,symmetries,operator)) in enumerate(zip(models,params,symmetry_list,operators))
    es = []
    spatial_symmetries = (i < 3) ? [nothing, "C4v"] : [nothing]
    for symmetry in spatial_symmetries
        for model_symmetry in symmetries
        kw = (i < 3) ? (spin_symmetry = model_symmetry,) : ()
            Ts_base = [Float64, BigFloat]
            Ts = isnothing(symmetry) ? Ts_base : complex.(Ts_base)
            for T = Ts
                for p = ps
                    @info "$(model), $(symmetry), $(model_symmetry), $(T), p=$(p)"
                    ce_alg = model(param...; kw..., p, T, symmetry)
                    if i < 3
                        observables = PEPO_observables([operator(kw[1])], ctm_alg)
                    else
                        observables = PEPO_observables([operator()], ctm_alg)
                    end
                    O = evolution_operator(ce_alg, β)
                    energies = ClusterExpansions.calculate_observables(O, χenv, observables)[1]
                    # observable = (O,i) -> ClusterExpansions.calculate_observables(O, χenv, observables)
                    # βs, expvals, = time_evolve(ce_alg, time_alg, trunc_alg, observable)

                    # energies = [e[1] for e in expvals]
                    @info "energies = $(energies)"

                    # @assert norm(imag.(energies)) < 1e-10
                    if norm(imag.(energies)) > 1e-10
                        @warn "imag = $(norm(imag.(energies)))"
                    end
                    push!(es, real.(energies))
                end
            end
        end
    end
    hi = reduce((a, b) -> max.(a, b), es)
    lo = reduce((a, b) -> min.(a, b), es)
    spread = hi .- lo
    @info "spread = $(spread)"
    @test maximum(spread) < 5e-5
end

