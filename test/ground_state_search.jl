using TensorKit
using TensorKitTensors
using PEPSKit
using OptimKit
using ClusterExpansions
using Test
using Random

Random.seed!(64468431)

D = 2
χenv = 20
T = ComplexF64

@testset "Ising model" begin
    (J, g, z) = (1.0, 2.5, 0.0)

    pspace = ℂ^2
    vspace = ℂ^D
    envspace = ℂ^χenv
    
    # Find the ground state with AD
    (Nr, Nc) = (1, 1)
    peps = InfinitePEPS(randn(T, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace'); unitcell = (Nr, Nc))
    H = transverse_field_ising(T, Trivial, InfiniteSquare(Nr, Nc); J, g)
    
    env0 = CTMRGEnv(peps, envspace)
    ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 2)
    env, = leading_boundary(env0, peps, ctm_alg)
    
    peps, env, E = fixedpoint(H, peps, env; optimizer_alg = LBFGS(8; gradtol = 1e-4, verbosity = 3, maxiter = 50), verbosity = 3);
    
    # Find the ground state with imaginary-time evolution
    ce_alg = ising_operators(J, g, z; T = Complex{BigFloat}, symmetry = "C4")
    time_alg = UniformGroundStateTimeEvolution(0.1, 0.1, 200, 1e-8; verbosity = 2)
    trunc_alg = NoEnvTruncation(truncdim(D); verbosity = 0)
    
    ctm_alg = SimultaneousCTMRG(; maxiter = 200, verbosity = 1)
    obss = PEPO_observables([H], ctm_alg)
    obs_function = O -> ClusterExpansions.calculate_observables(O, χenv, obss)
    
    A, E_CE = fixedpoint(ce_alg, time_alg, trunc_alg, obs_function);
    
    @test abs(E_CE[1] - E) < 5e-2
end

@testset "Spinless fermion model" begin
    (t, V, μ) = (1.0, -1.0, 0.0)
        
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    vspace = Vect[fℤ₂](0 => div(D,2), 1 => div(D,2))
    envspace = Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))
    
    # Find the ground state with AD
    (Nr, Nc) = (1, 1)
    peps = InfinitePEPS(randn(T, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace'); unitcell = (Nr, Nc))
    H = spinless_fermion_model(t, V, μ; T, Nr, Nc)
    
    env0 = CTMRGEnv(peps, envspace)
    ctm_alg = SimultaneousCTMRG(; maxiter = 250, verbosity = 2)
    env, = leading_boundary(env0, peps, ctm_alg)
    
    peps, env, E = fixedpoint(H, peps, env; optimizer_alg = LBFGS(4; gradtol = 1e-4, verbosity = 3, maxiter = 25), verbosity = 3);
    
    # Find the ground state with imaginary-time evolution
    ce_alg = spinless_fermion_operators(t, V, μ; T = Complex{BigFloat}, symmetry = "C4")
    time_alg = UniformGroundStateTimeEvolution(0.1, 0.1, 200, 1e-8; verbosity = 2)
    trunc_alg = NoEnvTruncation(truncdim(D); verbosity = 0)
    
    ctm_alg = SimultaneousCTMRG(; maxiter = 100, verbosity = 1)
    obss = PEPO_observables([H], ctm_alg)
    obs_function = O -> ClusterExpansions.calculate_observables(O, χenv, obss)
    
    A, E_CE = fixedpoint(ce_alg, time_alg, trunc_alg, obs_function);
    
    @test abs(E_CE[1] - E) < 2e-2
end