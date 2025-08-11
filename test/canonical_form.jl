using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Random
using Test

Random.seed!(489489)

@testset "Ising model PEPO" begin
    β = 0.4
    A = evolution_operator(ising_operators(1.0, 2.5, 0.0), β)

    tol_reconstruction = 1e-10
    canoc_alg = Canonicalization(; tol_canonical = 1e-7, tol_reconstruction, verbosity = 0)
    A′ = canonicalize(A, canoc_alg)

    pspace = ℂ^2
    χenv = 16
    envspace = ℂ^χenv

    M1 = SpinOperators.SS()
    M2 = SpinOperators.SᶻSᶻ()
    H1 = localoperator_model(pspace, M1)
    H2 = localoperator_model(pspace, M2)

    ctm_alg = SimultaneousCTMRG(; maxiter = 750, verbosity = 0)

    network = PEPSKit.trace_out(InfinitePEPO(A))
    env, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)
    h1 = expectation_value(InfinitePEPO(A), H1, env)
    h2 = expectation_value(InfinitePEPO(A), H2, env)

    network′ = PEPSKit.trace_out(InfinitePEPO(A′))
    env′, = leading_boundary(CTMRGEnv(network′, envspace), network′, ctm_alg)
    h1′ = expectation_value(InfinitePEPO(A′), H1, env′)
    h2′ = expectation_value(InfinitePEPO(A′), H2, env′)

    @test abs(h1-h1′) / abs(h1) < 10*tol_reconstruction
    @test abs(h2-h2′) / abs(h2) < 10*tol_reconstruction
end

@testset "Spinless fermion model PEPO" begin
    β = 0.4
    A = evolution_operator(spinless_fermion_operators(1.0, -1.0, 0.0), β)

    tol_reconstruction = 1e-10
    canoc_alg = Canonicalization(; tol_canonical = 1e-7, tol_reconstruction, verbosity = 0)
    A′ = canonicalize(A, canoc_alg)

    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    χenv = 16
    envspace = Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))

    M1 = FermionOperators.f_hop()
    M2 = FermionOperators.f⁻f⁺()
    H1 = localoperator_model(pspace, M1)
    H2 = localoperator_model(pspace, M2)

    ctm_alg = SimultaneousCTMRG(; maxiter = 750, verbosity = 0)

    network = PEPSKit.trace_out(InfinitePEPO(A))
    env, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)
    h1 = expectation_value(InfinitePEPO(A), H1, env)
    h2 = expectation_value(InfinitePEPO(A), H2, env)

    network′ = PEPSKit.trace_out(InfinitePEPO(A′))
    env′, = leading_boundary(CTMRGEnv(network′, envspace), network′, ctm_alg)
    h1′ = expectation_value(InfinitePEPO(A′), H1, env′)
    h2′ = expectation_value(InfinitePEPO(A′), H2, env′)

    @test abs(h1-h1′) / abs(h1) < 10*tol_reconstruction
    @test abs(h2-h2′) / abs(h2) < 10*tol_reconstruction
end

@testset "Spinless fermion model PEPS" begin
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    vspace = Vect[fℤ₂](0 => 3, 1 => 2)
    A = randn(ComplexF64, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')

    tol_reconstruction = 1e-10
    canoc_alg = Canonicalization(; tol_canonical = 1e-7, tol_reconstruction, verbosity = 1)
    A′ = canonicalize(A, canoc_alg)

    χenv = 30
    envspace = Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))

    M1 = FermionOperators.f_hop()
    M2 = FermionOperators.f⁻f⁺()
    H1 = localoperator_model(pspace, M1)
    H2 = localoperator_model(pspace, M2)

    ctm_alg = SimultaneousCTMRG(; maxiter = 750, verbosity = 2)

    network = InfinitePEPS(A)
    env, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)
    h1 = expectation_value(network, H1, env)
    h2 = expectation_value(network, H2, env)

    network′ = InfinitePEPS(A′)
    env′, = leading_boundary(CTMRGEnv(network′, envspace), network′, ctm_alg)
    h1′ = expectation_value(network′, H1, env′)
    h2′ = expectation_value(network′, H2, env′)

    @test abs(h1-h1′) / abs(h1) < 1e-3 #< 10*tol_reconstruction
    @test abs(h2-h2′) / abs(h2) < 1e-3 #< 10*tol_reconstruction
end

@testset "Ising model PEPS" begin
    pspace = ℂ^2
    vspace = ℂ^3
    A = randn(ComplexF64, pspace, vspace ⊗ vspace ⊗ vspace' ⊗ vspace')

    tol_reconstruction = 1e-10
    canoc_alg = Canonicalization(; tol_canonical = 1e-7, tol_reconstruction, verbosity = 1)
    A′ = canonicalize(A, canoc_alg)

    χenv = 24
    envspace = ℂ^χenv

    M1 = SpinOperators.SS()
    M2 = SpinOperators.SᶻSᶻ()
    H1 = localoperator_model(pspace, M1)
    H2 = localoperator_model(pspace, M2)

    ctm_alg = SimultaneousCTMRG(; maxiter = 750, verbosity = 2)

    network = InfinitePEPS(A)
    env, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)
    h1 = expectation_value(network, H1, env)
    h2 = expectation_value(network, H2, env)

    network′ = InfinitePEPS(A′)
    env′, = leading_boundary(CTMRGEnv(network′, envspace), network′, ctm_alg)
    h1′ = expectation_value(network′, H1, env′)
    h2′ = expectation_value(network′, H2, env′)

    @test abs(h1-h1′) / abs(h1) < 1e-3 #< 10*tol_reconstruction
    @test abs(h2-h2′) / abs(h2) < 1e-3 #< 10*tol_reconstruction
end

