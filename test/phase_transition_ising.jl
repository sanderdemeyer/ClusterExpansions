using TensorKit
using TensorKitTensors
using MPSKit
using ClusterExpansions
using PEPSKit
using Plots
using Random

Random.seed!(1654841489)

# Set up time evolution algorithm
β₀ = 0.1
Δβ = 0.1
maxiter = 8
time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)

χenv = 8 # Environment bond dimension used in the calculation of expectation values

# Set up truncation algorithm
Dcut = 4
trunc_alg = NoEnvTruncation(truncdim(Dcut); verbosity = 0)

# Define observables
vumps_alg = VUMPS(; maxiter = 100, verbosity = 0)
obss = PEPO_observables([:spectrum, SpinOperators.σᶻ(), SpinOperators.σˣ()], vumps_alg)
obs_function = (O,i) -> ClusterExpansions.calculate_observables(O, χenv, obss)

@testset "Classical Ising model" begin
    # Set up the classical Ising model
    (J, g, z) = (1.0, 0.0, 0.0)
    ce_alg = ising_operators(J, g, z; T = Complex{BigFloat}, symmetry = "C4")

    # Perform the time evolution.
    βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, obs_function)

    # Extract the expectation values
    ξs = [e[1][1] for e in expvals]
    mzs = [e[2] for e in expvals]
    mxs = [e[3] for e in expvals]

    # Critical temperature for the classical Ising model
    Tc = 2/(log(1+sqrt(2)))
    βc = 1 / Tc

    # Tests on the phase transition of the classical Ising model
    @test norm(mxs) < 1e-14
    @test all([((β < βc) && (abs(mz) < 0.5)) || ((β > βc) && (abs(mz) > 0.5)) for (β,mz) in zip(βs,mzs)])
end

@testset "Quantum Ising model" begin
    # Set up the classical Ising model
    (J, g, z) = (1.0, 2.5, 0.0)
    ce_alg = ising_operators(J, g, z; T = Complex{BigFloat}, symmetry = "C4")

    # Perform the time evolution.
    βs, expvals, Os = time_evolve(ce_alg, time_alg, trunc_alg, obs_function)

    # Extract the expectation values
    ξs = [e[1][1] for e in expvals]
    mzs = [e[2] for e in expvals]
    mxs = [e[3] for e in expvals]

    # Critical temperature for the classical Ising model
    Tc = 1.2737
    βc = 1 / Tc

    # Tests on the phase transition of the classical Ising model    
    @test all([((β < βc) && (abs(mz) < 0.5)) || ((β > βc) && (abs(mz) > 0.5)) || (abs(β-βc) < 2e-2) for (β,mz) in zip(βs,mzs)])
end
