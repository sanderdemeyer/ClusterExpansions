using TensorKit
using TensorKitTensors
using PEPSKit
using MPSKit
using ClusterExpansions
using Test
using Random
using LinearAlgebra

const CE = ClusterExpansions

Random.seed!(2718281828)

T = Float64
unitcell = (2, 2, 2)
pspace = ℂ^2

@testset "Structure and weight bookkeeping" begin
    peps = InfiniteWeightPEPS3D(randn, T, pspace, ℂ^2; unitcell)
    @test size(peps) == unitcell

    site = CartesianIndex(1, 2, 1)
    alldirs = ntuple(identity, 6)
    allfalse = ntuple(Returns(false), 6)
    A0 = peps.vertices[site]
    A1 = CE._absorb_weights3D(A0, peps.weights, site, alldirs, allfalse, false)
    A2 = CE._absorb_weights3D(A1, peps.weights, site, alldirs, allfalse, true)
    @test norm(A2 - A0) < 1e-12

    # splitting off the reduced tensors on a bond and gluing them back is the identity
    for axis in 1:3
        dir = CE._posdir3D(axis)
        site2 = CE._shift_site(site, axis, 1, unitcell)
        A, B = peps.vertices[site], peps.vertices[site2]
        X, a, b, Y = CE._qr_bond3D(A, B, dir)
        A′, B′ = CE._qr_bond_undo3D(X, a, b, Y, dir)
        @test norm(A′ - A) < 1e-12
        @test norm(B′ - B) < 1e-12
    end

    # reduced density matrices are normalized and hermitian
    ρ1 = onesite_rdm(peps, site)
    @test tr(ρ1) ≈ 1
    @test norm(ρ1 - ρ1') < 1e-12
    for axis in 1:3
        ρ2 = bond_rdm(peps, site, axis)
        @test tr(ρ2) ≈ 1
        @test norm(ρ2 - ρ2') < 1e-12
    end
end

@testset "Bond update applies the gate exactly" begin
    # with an untruncated bond, one update must reproduce the exact action of the gate on
    # the two-site wavefunction, for every axis and for non-trivial bond weights
    D = 2
    peps = InfiniteWeightPEPS3D(randn, T, pspace, ℂ^D; unitcell)
    for I in CartesianIndices(peps.weights.data)
        peps.weights[I] = DiagonalTensorMap(0.2 .+ rand(D), ℂ^D)
    end
    h = TensorKitTensors.SpinOperators.S_exchange(T, Trivial; spin=1//2)
    gate = exp(-0.3 * h)

    function theta(peps, site, axis)
        dir, dirop = CE._posdir3D(axis), CE._opposite3D(CE._posdir3D(axis))
        site2 = CE._shift_site(site, axis, 1, size(peps))
        alldirs = ntuple(identity, 6)
        A = CE._absorb_weights3D(
            peps.vertices[site], peps.weights, site, alldirs, ntuple(==(dir), 6), false
        )
        B = CE._absorb_weights3D(
            peps.vertices[site2], peps.weights, site2, alldirs, ntuple(==(dirop), 6), false
        )
        idxA, idxB = Vector{Int}(undef, 7), Vector{Int}(undef, 7)
        idxA[1], idxB[1] = -1, -2
        n = 2
        for d in 1:6
            idxA[1 + d] = (d == dir) ? 1 : (n += 1; -n)
        end
        for d in 1:6
            idxB[1 + d] = (d == dirop) ? 1 : (n += 1; -n)
        end
        return permute(ncon([A, B], [idxA, idxB]), ((1, 2), Tuple(3:12)))
    end

    site = CartesianIndex(1, 1, 1)
    for axis in 1:3
        p = deepcopy(peps)
        θ0 = theta(p, site, axis)
        @tensor θe[-1 -2; -3 -4 -5 -6 -7 -8 -9 -10 -11 -12] :=
            gate[-1 -2; 1 2] * θ0[1 2; -3 -4 -5 -6 -7 -8 -9 -10 -11 -12]
        alg = SimpleUpdate3D(0.3, 1e-10, 1, notrunc(); normalize_vertices=false)
        ϵ = CE._su_bond!(site, axis, gate, p, alg)
        θ1 = theta(p, site, axis)
        @test ϵ < 1e-12
        @test norm(θ1 / norm(θ1) - θe / norm(θe)) < 1e-12
    end
end

@testset "Exactly solvable limits" begin
    D = 2
    # zero coupling: the ground state of -g Σ σˣ is the |+⟩ product state with E = -g
    g = 1.0
    H = CubicHamiltonian(
        rmul!(PEPSKit.σᶻᶻ(T, Trivial), 0.0), rmul!(PEPSKit.σˣ(T, Trivial), -g); unitcell
    )
    peps = InfiniteWeightPEPS3D(randn, T, pspace, ℂ^D; unitcell)
    peps, = simpleupdate3D(
        peps, H, [0.1, 0.01]; tol=1e-13, maxiter=300, trscheme=truncrank(D), verbosity=0
    )
    @test real(energy_per_site(peps, H)) ≈ -g atol = 1e-8
    @test real(expectation_value(peps, PEPSKit.σˣ(T, Trivial))) ≈ 1 atol = 1e-8

    # zero field: the classical Ising ground state has E = -z/2 = -3 per site
    H = CubicHamiltonian(rmul!(PEPSKit.σᶻᶻ(T, Trivial), -1.0); unitcell)
    peps = InfiniteWeightPEPS3D(randn, T, pspace, ℂ^D; unitcell)
    peps, = simpleupdate3D(
        peps, H, [0.2, 0.05, 0.01]; tol=1e-13, maxiter=400, trscheme=truncrank(D), verbosity=0
    )
    @test real(energy_per_site(peps, H)) ≈ -3 atol = 1e-8
    @test abs(real(expectation_value(peps, PEPSKit.σᶻ(T, Trivial)))) ≈ 1 atol = 1e-8
end

@testset "Transverse-field Ising against perturbation theory" begin
    # deep in the disordered phase, E/site = -g - 3J²/(4g) + O(g^-3)
    plus = TensorMap(T[1, 1] / sqrt(2), pspace ← one(pspace))
    for g in (6.0, 10.0)
        H = CubicHamiltonian(
            rmul!(PEPSKit.σᶻᶻ(T, Trivial), -1.0), rmul!(PEPSKit.σˣ(T, Trivial), -g); unitcell
        )
        peps = product_peps3D(plus, ℂ^3; unitcell, noise=1e-1)
        peps, = simpleupdate3D(
            peps, H, [0.05, 0.01, 1e-3]; tol=1e-13, maxiter=400, verbosity=0
        )
        @test real(energy_per_site(peps, H)) ≈ -g - 3 / (4g) rtol = 5e-4
    end
end

@testset "Heisenberg antiferromagnet on the cubic lattice" begin
    # QMC: E/site = -0.9025 J, staggered magnetization 0.4227; simple update slightly
    # underestimates the energy gain and overestimates the order parameter
    h = TensorKitTensors.SpinOperators.S_exchange(T, Trivial; spin=1//2)
    Sz = TensorKitTensors.SpinOperators.S_z(T, Trivial; spin=1//2)
    H = CubicHamiltonian(h; unitcell)
    up = TensorMap(T[1, 0], pspace ← one(pspace))
    dn = TensorMap(T[0, 1], pspace ← one(pspace))
    neel = [iseven(x + y + z) ? up : dn for x in 1:2, y in 1:2, z in 1:2]
    Vs = fill(ℂ^2, unitcell)
    peps = product_peps3D(neel, Vs, Vs, Vs; noise=1e-1)
    peps, = simpleupdate3D(
        peps, H, [0.1, 0.03, 0.01, 1e-3]; tol=1e-13, maxiter=400, verbosity=0
    )
    E = real(energy_per_site(peps, H))
    ms = sum(CartesianIndices(peps.vertices)) do site
        return (-1)^sum(Tuple(site)) * real(tr(onesite_rdm(peps, site) * Sz))
    end / length(peps.vertices)
    @test -0.92 < E < -0.87
    @test 0.42 < ms < 0.47
    # all three axes are equivalent
    bonds = [real(tr(bond_rdm(peps, CartesianIndex(1, 1, 1), ax) * h)) for ax in 1:3]
    @test maximum(bonds) - minimum(bonds) < 1e-4
end

@testset "Weight absorption round trip" begin
    peps = InfiniteWeightPEPS3D(randn, T, pspace, ℂ^2; unitcell)
    peps2 = InfiniteWeightPEPS3D(InfinitePEPS3D(peps))
    # absorbing all weights and re-attaching trivial ones leaves the physical state invariant
    for site in CartesianIndices(peps.vertices)
        @test norm(onesite_rdm(peps, site) - onesite_rdm(peps2, site)) < 1e-10
    end
end
