using TensorKit
using TensorKitTensors
using ClusterExpansions
using Test
using Random

Random.seed!(1654843513)
setprecision(128)

@testset "Symmetries of fermionic cluster expansion" begin
    (t, V, μ) = (1.0, -2.5, 0.0)
    β = 0.1
    T = Complex{Float64}

    ce_alg = spinless_fermion_operators(t, V, μ; T, symmetry = nothing)
    O_symm, _ = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    ce_alg = spinless_fermion_operators(t, V, μ; T, symmetry = "C4")
    O_asymm, _ = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)

    for (key, value) in O_symm
        @test norm(O_symm[key])-norm(O_asymm[key]) < 1e-14
    end
end

@testset "Loop solver" begin
    T = Complex{BigFloat}
    β = 1e-2
    symmetry = "C4"

    twosite_op = FermionOperators.f_hop()
    onesite_op = FermionOperators.f_num()

    cluster = ClusterExpansions.Cluster([(0,0), (1,0), (1,1), (0,1)])

    exp_H = ClusterExpansions.exponentiate_hamiltonian(twosite_op, onesite_op, cluster, β)
    levels_to_update = [(-1, -1, 0, 0) (0, -1, -1, 0) (0, 0, -1, -1) (-1, 0, 0, -1)]
    spaces = i -> Vect[fℤ₂](0 => 3, 1 => 3)

    As_asymm, error_asymm = ClusterExpansions.solve_4_loop_optim(exp_H, spaces, levels_to_update; verbosity = 0, symmetry = nothing);
    As_symm, error_symm = ClusterExpansions.solve_4_loop_optim(exp_H, spaces, levels_to_update; verbosity = 0, symmetry = "C4");

    @test error_asymm < 1e-3
    @test error_symm < 2e-2
    @test norm(ClusterExpansions.rotl90_fermionic(As_symm[1])-As_symm[4]) < 1e-14
end