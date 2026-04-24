using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots

t = 1.0
J = 0.5
μ = 1.0

χenv = 20 # Environment bond dimension used in the calculation of expectation values

# Parameters in the truncation scheme
Dcut = 16
trunc_alg = NoEnvTruncation(truncdim(Dcut); verbosity = 1)

# Set up time evolution algorithm
β₀ = 0.0
Δβ = 0.05
maxiter = 80
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)

ctm_alg = SimultaneousCTMRG(; maxiter = 500)
vumps_alg = VUMPS(; maxiter = 250, verbosity = 1)

trunc_alg_start = NoEnvTruncation(truncdim(9))

ns = []
for particle_symmetry = [Trivial, U1Irrep]
    for spin_symmetry = [Trivial, U1Irrep, SU2Irrep]

        # observables = PEPO_observables([TJOperators.S_z(particle_symmetry, spin_symmetry), TJOperators.e_num(particle_symmetry, spin_symmetry)], ctm_alg)
        observables = PEPO_observables([TJOperators.e_num(particle_symmetry, spin_symmetry)], ctm_alg)
        observable = (O,i) -> ClusterExpansions.calculate_observables(O, χenv, observables)
        observable_during = (O,i) -> []

        ce_alg = tJ_operators(t, J, μ; h, T = Float64, particle_symmetry, spin_symmetry, symmetry = nothing, p = 3, verbosity = 0, svd = true)

        βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, observable_during; trunc_alg_start, normalizing = true);

        n = observable(Os[end], 1)[1]
        push!(ns, n)
        # n_final = observable(Os[end], 1)[2]
    end
end