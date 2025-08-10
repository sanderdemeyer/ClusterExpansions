using TensorKit
using TensorKitTensors
using KrylovKit
using ClusterExpansions
using PEPSKit
using Test
using Random

Random.seed!(5849032580)

function test_correlation_functions(pspace, vspace, envspace, operators; R = 5)
    (Nr, Nc) = (1, 1)
    pspaces = fill(pspace, Nr, Nc)

    ctm_alg = SimultaneousCTMRG(; maxiter = 300)
    peps = InfinitePEPS(rand, Float64, pspace, vspace; unitcell=(1,1));
    env, = leading_boundary(CTMRGEnv(peps, envspace), peps, ctm_alg);

    site0 = CartesianIndex(1, 1)
    site1s = collect(site0 + CartesianIndex(0, i) for i in 1:R)

    for M = operators

        corr_efficient = correlator_horizontal(peps, peps, env, M, (CartesianIndex(1,1), CartesianIndex(1,1+R)))
        corr = collect(
            begin
                O = PEPSKit.LocalOperator(pspaces, (site0, site1) => M)
                val = expectation_value(peps, O, env)
            end for site1 in site1s
        )
        @test norm(corr_efficient - corr) < 1e-14
    end
end

@testset "Ising model" begin
    pspace = ℂ^2
    vspace = ℂ^4
    envspace = ℂ^20

    operators = [SpinOperators.SS(), SpinOperators.σˣ() ⊗ SpinOperators.σˣ()]

    test_correlation_functions(pspace, vspace, envspace, operators)
end

@testset "Spinless fermion model" begin
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    vspace = Vect[fℤ₂](0 => 2, 1 => 2)

    χenv = 20
    envspace = Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))

    operators = [FermionOperators.f_hop(), FermionOperators.f_num() ⊗ FermionOperators.f_num()]
    
    test_correlation_functions(pspace, vspace, envspace, operators)
end