using TensorKit
using PEPSKit
using ClusterExpansions
using Test
using LinearAlgebra

const CE = ClusterExpansions

# For the classical Ising model H = -Σ J_a Z_i Z_j - h Σ Z_i on the cubic lattice, all bond
# terms commute and every bond gate e^{K ZZ} = cosh K + sinh K ZZ has operator-Schmidt rank 2.
# The density operator ρ(β) = e^{-βH} is therefore *exactly* a D = 2 PEPO at every β, and
# finite-temperature simple update with D = 2, started from ρ ∝ 𝟙, can never truncate
# anything: it must reproduce ρ(β) exactly, independently of the time step.
#
# This is checked in a gauge-invariant way by contracting the 2×2×2 unit cell on a periodic
# 2×2×2 torus and comparing all 2^8 diagonal elements with the exact Boltzmann weights of the
# Ising model on that torus.

T = ComplexF64
unitcell = (2, 2, 2)
P = ℂ^2
dm = DensityMatrixSpace(T, P)
sites = CartesianIndices(unitcell)

function classical_ising(J::NTuple{3,Real}, h::Real)
    zz = PEPSKit.σᶻᶻ(T, Trivial)
    twosite = map(Ja -> rmul!(copy(zz), -Ja), J)
    onesite = iszero(h) ? nothing : rmul!(PEPSKit.σᶻ(T, Trivial), -h)
    return ThermalCubicHamiltonian(dm, twosite, onesite; unitcell)
end

# exact starting point ρ ∝ 𝟙 embedded in D = 2 (no noise, so no junk entanglement)
initial_peps() = infinite_temperature_peps3D(dm, ℂ^2; unitcell, noise=0.0)

function run_su(H, β, nsteps)
    peps, = simpleupdate3D(
        initial_peps(), H, [β / nsteps]; tol=-Inf, maxiter=nsteps, verbosity=0
    )
    return peps
end

# unique positive contraction label for the bond `(axis, site)`
bondlabel(axis, site) = 3 * (LinearIndices(unitcell)[site] - 1) + axis

"""
Contract the PEPO on the periodic 2×2×2 torus, keeping the diagonal physical index of every
site open. Returns the normalized array of weights `W[s_1, ..., s_8]`.
"""
function torus_weights(peps)
    pepo = cubic_pepo(peps, dm)
    tensors, indices = Array{T}[], Vector{Int}[]
    for site in sites
        O = convert(Array, pepo[site])  # (ket, bra, U, N, E, D, S, W)
        push!(tensors, cat([O[s, s, ntuple(_ -> :, 6)...] for s in 1:2]...; dims=7))
        neg(axis) = bondlabel(axis, CE._shift_site(site, axis, -1, unitcell))
        push!(
            indices,
            [
                bondlabel(3, site),
                bondlabel(2, site),
                bondlabel(1, site),
                neg(3),
                neg(2),
                neg(1),
                -LinearIndices(unitcell)[site],
            ],
        )
    end
    W = ncon(tensors, indices)
    return W / sum(W)
end

"""Largest off-diagonal physical component of the site tensors, relative to their norm."""
function offdiagonal(peps)
    return maximum(cubic_pepo(peps, dm)) do O
        Oarr = convert(Array, O)
        offdiag = (Oarr[1, 2, ntuple(_ -> :, 6)...], Oarr[2, 1, ntuple(_ -> :, 6)...])
        return norm(norm.(offdiag)) / norm(Oarr)
    end
end

"""Exact Boltzmann weights of the classical Ising model on the periodic 2×2×2 torus."""
function exact_torus_weights(J, h, β)
    W = zeros(Float64, ntuple(_ -> 2, length(sites)))
    for conf in CartesianIndices(W)
        s = [3 - 2 * conf[LinearIndices(unitcell)[site]] for site in sites]  # 1 ↦ +1, 2 ↦ -1
        E = 0.0
        for site in sites
            i = LinearIndices(unitcell)[site]
            E -= h * s[i]
            for axis in 1:3
                j = LinearIndices(unitcell)[CE._shift_site(site, axis, 1, unitcell)]
                E -= J[axis] * s[i] * s[j]
            end
        end
        W[conf] = exp(-β * E)
    end
    return W / sum(W)
end

@testset "D = 2 simple update is exact for the classical Ising model" begin
    for (J, h, β) in [
        ((1.0, 1.0, 1.0), 0.0, 0.2216546),  # isotropic, at the 3D critical point
        ((1.0, 1.0, 1.0), 0.0, 0.5),        # isotropic, deep in the ordered phase
        ((1.0, 0.7, 0.4), 0.0, 0.3),        # anisotropic couplings
        ((1.0, 0.7, 0.4), 0.3, 0.3),        # anisotropic couplings + longitudinal field
    ]
        @testset "J = $J, h = $h, β = $β" begin
            H = classical_ising(J, h)
            Wexact = exact_torus_weights(J, h, β)
            # commuting gates: no Trotter error, so a coarse and a fine time step must agree
            for nsteps in (5, 200)
                peps = run_su(H, β, nsteps)
                @test all(w -> space(w, 1) == ℂ^2, peps.weights)
                @test offdiagonal(peps) < 1e-12  # ρ stays diagonal
                W = torus_weights(peps)
                @test maximum(abs.(imag.(W))) < 1e-10
                @test maximum(abs.(real.(W) - Wexact)) / maximum(Wexact) < 1e-10
            end
        end
    end
end

# The exact D = 2 PEPO is in canonical (simple-update) form with Schmidt weights
# (cosh βJ_a, sinh βJ_a) ∝ (1, tanh βJ_a). The bond weights are a gauge choice, however, and
# although the contracted ρ stays exact (see above), this canonical gauge is only a stable
# fixed point of the simple-update sweep for β < β* = atanh(1/5) / 2 ≈ 0.101: the weights are
# belief-propagation messages of the doubled network ⟨ρ|ρ⟩ = Tr e^{-2βH}, whose paramagnetic
# fixed point on the z = 6 cubic lattice destabilizes once (z - 1) tanh(2β) > 1. Beyond β*,
# rounding errors in the gauge are amplified every sweep, and after enough sweeps the weights
# drift far away from (1, tanh βJ_a) without affecting the represented ρ.
canonical_weights(peps, J, β; atol=1e-10) = all(
    isapprox(diag(convert(Array, peps.weights[axis, site])), [1.0, tanh(β * J[axis])]; atol)
    for site in sites, axis in 1:3
)

@testset "Bond weights of the D = 2 classical Ising PEPO" begin
    J = (1.0, 0.7, 0.4)
    H = classical_ising(J, 0.0)
    # below β*: canonical gauge is stable, even after many sweeps
    @test canonical_weights(run_su(H, 0.09, 400), J, 0.09)
    # above β*: fine as long as rounding errors have too few sweeps to be amplified ...
    @test canonical_weights(run_su(H, 0.3, 5), J, 0.3)
    # ... but the gauge drifts away after many sweeps (ρ itself remains exact)
    @test_broken canonical_weights(run_su(H, 0.3, 200), J, 0.3; atol=1e-6)
end

@testset "D = 2 simple update never truncates for the classical Ising model" begin
    H = classical_ising((1.0, 0.7, 0.4), 0.3)
    dt = 0.05
    gates = CE.get_expham(dt / 2, H)
    alg = SimpleUpdate3D(dt, -Inf, 1, FixedSpaceTruncation())
    peps = initial_peps()
    for _ in 1:10
        peps, ϵ = su_iter3D(gates, peps, alg)
        @test ϵ < 1e-10
    end
end

#
## Thermal observables directly from the simple-update bond weights
#
# A density-operator PEPS evolved to imaginary time τ is a purification of the thermal state
# at β = 2τ, and the (canonicalized) bond weights are the belief-propagation environment of
# its norm network. For the classical Ising model with D = 2, ρ is exact, so the SU estimate
# must equal the Bethe approximation; on a 1D chain (a tree) it must even be exact.

"""Bethe approximation `(⟨Z⟩, ⟨ZZ⟩, energy per site)` on a lattice with coordination `z`."""
function bethe_ising(J, h, β; z=6)
    t = tanh(β * J)
    x = 1.0  # cavity field (times β), started on the h ≥ 0 ferromagnetic branch
    for _ in 1:10_000
        x = β * h + (z - 1) * atanh(t * tanh(x))
    end
    m = tanh(β * h + z * atanh(t * tanh(x)))
    c = (t + tanh(x)^2) / (1 + t * tanh(x)^2)
    return m, c, -J * z / 2 * c - h * m
end

"""Exact `(⟨Z⟩, ⟨ZZ⟩)` of the infinite 1D classical Ising chain, via its transfer matrix."""
function chain_ising(J, h, β)
    S = [1.0, -1.0]
    TM = [exp(β * J * a * b + β * h * (a + b) / 2) for a in S, b in S]
    F = eigen(Symmetric(TM))
    λ, v = F.values[end], F.vectors[:, end]
    return sum(v .^ 2 .* S), (v .* S)' * TM * (v .* S) / λ
end

"""Run SU up to τ = β/2, so that the SU observables are those at inverse temperature β."""
function thermal_su(H, β, Vs...; nsteps=100)
    peps = infinite_temperature_peps3D(dm, Vs...; unitcell, noise=0.0)
    peps, = simpleupdate3D(peps, H, [β / 2 / nsteps]; tol=-Inf, maxiter=nsteps, verbosity=0)
    return canonicalize3D(peps)
end

@testset "Simple-update thermal observables" begin
    sz = PEPSKit.σᶻ(T, Trivial)
    zz = PEPSKit.σᶻᶻ(T, Trivial)

    @testset "1D chain: exact (h = $h)" for h in (0.0, 0.3)
        β, J = 0.7, 1.0
        H = classical_ising((J, 0.0, 0.0), h)
        V2, V1 = fill(ℂ^2, unitcell), fill(ℂ^1, unitcell)
        peps = thermal_su(H, β, V2, V1, V1)
        mexact, cexact = chain_ising(J, h, β)
        @test real(expectation_value(peps, dm, sz)) ≈ mexact atol = 1e-10
        for site in sites
            @test real(tr(bond_rdm(peps, dm, site, 1) * zz)) ≈ cexact atol = 1e-10
        end
        @test real(energy_per_site(peps, H)) ≈ -J * cexact - h * mexact atol = 1e-10
    end

    @testset "3D: Bethe approximation (β = $β, h = $h)" for (β, h) in [
        (0.15, 0.0),  # paramagnetic, below β_Bethe = atanh(1/5) ≈ 0.2027
        (0.3, 0.2),   # field-polarized
        (0.5, 0.2),
    ]
        J = 1.0
        H = classical_ising((J, J, J), h)
        peps = thermal_su(H, β, ℂ^2)
        m, c, E = bethe_ising(J, h, β)
        @test real(expectation_value(peps, dm, sz)) ≈ m atol = 1e-10
        @test real(expectation_value(peps, dm, zz)) ≈ c atol = 1e-10
        @test real(energy_per_site(peps, H)) ≈ E atol = 1e-10
    end

    @testset "canonicalize3D leaves ρ unchanged" begin
        J, h, β = (1.0, 0.7, 0.4), 0.3, 0.3
        H = classical_ising(J, h)
        peps = infinite_temperature_peps3D(dm, ℂ^2; unitcell, noise=0.0)
        peps, = simpleupdate3D(peps, H, [β / 50]; tol=-Inf, maxiter=50, verbosity=0)
        peps2 = canonicalize3D(peps)
        @test !(peps2.weights ≈ peps.weights)  # the gauge did change ...
        @test torus_weights(peps2) ≈ torus_weights(peps) rtol = 1e-10  # ... the state did not
    end
end
