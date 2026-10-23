using TensorKit
using TensorKitTensors
using PEPSKit
using MPSKit
using ClusterExpansions
using Test
using Random
using LinearAlgebra

const CE = ClusterExpansions

Random.seed!(31415926535)

T = ComplexF64
unitcell = (2, 2, 2)
P = ℂ^2

@testset "DensityMatrixSpace: vectorize / unvectorize" begin
    dm = DensityMatrixSpace(T, P)
    @test dim(dm.fused) == dim(P)^2

    O = randn(T, P ← P)
    v = vectorize(dm, O)
    @test numout(v) == 1 && numin(v) == 0
    @test norm(unvectorize(dm, v) - O) < 1e-12

    # for Hermitian O, vectorize(id) is the trace functional: <vec(id), vec(O)> = Tr[O]
    idvec = vectorize(dm, id(T, P))
    @test dot(idvec, v) ≈ tr(block(O, Trivial()))
end

@testset "thermal_gate implements ρ ↦ GρG† exactly" begin
    dm = DensityMatrixSpace(T, P)
    Ubig = dm.U ⊗ dm.U

    function vec2site(O)
        O4 = permute(O, ((1, 2, 3, 4), ()))
        O4 = permute(O4, ((1, 3, 2, 4), ()))
        return Ubig' * O4
    end

    h = randn(T, P ⊗ P ← P ⊗ P)
    h = (h + h') / 2
    dt = 0.37
    g = exp((-dt / 2) * h)

    rho = randn(T, P ⊗ P ← P ⊗ P)
    rho = permute(rho * rho', ((1, 2), (3, 4)))  # positive semi-definite, like a real ρ

    v_exact = vec2site(g * rho * g')
    v_super = CE.thermal_gate(dt, h, dm) * vec2site(rho)
    @test norm(v_super - v_exact) / norm(v_exact) < 1e-12
end

@testset "Finite-T simple update reproduces exact single-site thermal state" begin
    # H = -g σˣ (no coupling): ρ = e^{-βH} factorizes exactly, D=1 throughout, so the
    # infinite-temperature seed + thermal gate must reproduce the single-site thermal state
    # exactly (Z ∝ 2cosh(βg), ⟨σˣ⟩ = tanh(βg)).
    g = 1.3
    dm = DensityMatrixSpace(T, P)
    onesite = rmul!(PEPSKit.σˣ(T, Trivial), -g)
    H = ThermalCubicHamiltonian(dm, rmul!(PEPSKit.σᶻᶻ(T, Trivial), 0.0), onesite; unitcell)
    peps = infinite_temperature_peps3D(dm, ℂ^1; unitcell, noise=0.0)

    β = 0.734
    peps, = simpleupdate3D(peps, H, [β / 200]; tol=-Inf, maxiter=200, verbosity=0)

    site = CartesianIndex(1, 1, 1)
    v = peps.vertices[site]
    for i in 7:-1:2
        v = removeunit(v, i)
    end
    rho = unvectorize(dm, v)
    sx = PEPSKit.σˣ(T, Trivial)

    Z = tr(block(rho, Trivial()))
    m = tr(block(rho * sx, Trivial())) / Z
    @test real(m) ≈ tanh(β * g) atol = 1e-8

    # the `cubic_pepo` conversion (feeding into the existing ctmrg_cubic.jl machinery) must
    # give the same answer via the exact same "twist-trace" contraction pattern it uses.
    pepo = cubic_pepo(peps, dm)
    Otens = pepo[Tuple(site)...]
    @tensor Otr[-1 -2 -3 -4 -5 -6] := twist(Otens, 2)[1 1; -1 -2 -3 -4 -5 -6]
    @tensor Mtr[-1 -2 -3 -4 -5 -6] := twist(Otens, 2)[1 2; -1 -2 -3 -4 -5 -6] * sx[2; 1]
    Zpepo = tr(block(permute(Otr, ((), Tuple(1:6))), Trivial()))
    Mpepo = tr(block(permute(Mtr, ((), Tuple(1:6))), Trivial()))
    @test real(Mpepo / Zpepo) ≈ tanh(β * g) atol = 1e-8
end

@testset "Finite-T simple update reproduces exact classical-Ising bond correlator" begin
    # H = -J ZZ along x only (y, z decoupled): ρ = e^{-βH} is diagonal (commuting terms), so
    # the exact nearest-neighbour correlator of the classical/quantum-commuting 1D Ising
    # chain, ⟨Z_iZ_{i+1}⟩ = tanh(βJ), gives a nontrivial (entangled-bond) exact benchmark.
    J = 1.0
    dm = DensityMatrixSpace(T, P)
    zz = PEPSKit.σᶻᶻ(T, Trivial)
    zz0 = rmul!(copy(zz), 0.0)
    H = ThermalCubicHamiltonian(dm, (rmul!(copy(zz), -J), zz0, zz0); unitcell)

    D = 4
    Vx = fill(ℂ^D, unitcell)
    V1 = fill(ℂ^1, unitcell)
    peps = infinite_temperature_peps3D(dm, Vx, V1, V1; unitcell, noise=1e-2)

    β = 0.6
    peps, = simpleupdate3D(
        peps, H, [β / 400]; tol=-Inf, maxiter=400, verbosity=0, trscheme=FixedSpaceTruncation()
    )

    pepo = cubic_pepo(peps, dm)
    sz = PEPSKit.σᶻ(T, Trivial)
    idop = id(T, P)

    function reduced(Otens)  # drop the trivial U, N, D, S legs -> (P, P') ← (E, W)
        A = Otens
        A = removeunit(A, 7)
        A = removeunit(A, 6)
        A = removeunit(A, 4)
        A = removeunit(A, 3)
        return A
    end
    function transfer(Otens, obsop)
        Ar = reduced(Otens)
        @tensor M[w; e] := twist(Ar, 2)[p1 p2; e w] * obsop[p2; p1]
        return M
    end

    M1 = transfer(pepo[1, 1, 1], idop)
    M2 = transfer(pepo[2, 1, 1], idop)
    M1z = transfer(pepo[1, 1, 1], sz)
    M2z = transfer(pepo[2, 1, 1], sz)

    ringtrace(mats) = tr(block(reduce(*, mats), Trivial()))

    N = 40
    base = [iseven(i) ? M1 : M2 for i in 1:N]
    Z = ringtrace(base)
    mats = copy(base)
    mats[1], mats[2] = M1z, M2z
    ZZ = ringtrace(mats)

    @test real(ZZ / Z) ≈ tanh(β * J) atol = 1e-3
end
