using TensorKit
using TensorKitTensors
using ClusterExpansions
using PEPSKit
using Plots
using JLD2
using MPSKit

t = 1.0 # Nearest-neighbour hopping term
V = -2.5 # Interaction term. Repulsive if positive.
μ = 0.0 # Extra chemical potential on top of half-filling
@assert μ == 0.0 "mu set to zero to have no onsite interaction"
ctm_alg = SimultaneousCTMRG(; maxiter = 400, verbosity = 1)
(Nr, Nc) = (2, 2)

pspace = Vect[fℤ₂](0 => 1, 1 => 1)
T = Float64
kinetic_operator = FermionOperators.f_hop(T)
number_operator = FermionOperators.f_num(T)
number_operator_halffilling = number_operator - id(pspace)/2
symmetry_breaking_term = FermionOperators.f⁻f⁻(T) - FermionOperators.f⁺f⁺(T)
@tensor number_twosite[-1 -2; -3 -4] := number_operator_halffilling[-1; -3] * number_operator_halffilling[-2; -4]
twosite_op = rmul!(kinetic_operator, -T(t)) + rmul!(number_twosite, T(V)) #- rmul!(symmetry_breaking_term, T(δ))

function get_both_observables(O::InfinitePEPO, virtualspace, vumps_alg, M)
    pf = PEPSKit.trace_out(O)
    T = InfiniteMPO([pf[1,1]])

    pspace = domain(pf[1,1])[2]

    mps = InfiniteMPS([
        randn(
            ComplexF64,
            virtualspace * pspace,
            virtualspace,
        )])
    mps, env, _ = leading_boundary(mps, T, vumps_alg);
    ϵ, δ, = marek_gap(mps; num_vals = 20)
    
    t = O[1,1]
    E_num = PEPSKit.@autoopt @tensor twist(t,2)[d1 d2; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) * M[d2; d1] * 
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(t,2)[d d; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) *
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    return E_num / E_denom, 1 / ϵ

end

function convert_to_pepo_fuser(A, F)
    @tensor pepo[-1 -2; -3 -4 -5 -6] := A[1; -3 -4 -5 -6] * conj(F[1; -1 -2])
end

function get_data_SU(Dcut, χenv, dt)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)
    trivspace = Vect[fℤ₂](0 => 1)
    vspace = Vect[fℤ₂](0 => div(Dcut,2), 1 => div(Dcut,2))
    pspace_fused = fuse(pspace, pspace)
    truncspace = Vect[fℤ₂](0 => div(Dcut,2), 1 => div(Dcut,2))
    envspace = Vect[fℤ₂](0 => div(χenv,2), 1 => div(χenv,2))

    state0 = permute(id(pspace ⊗ trivspace ⊗ trivspace), ((1,4),(5,6,2,3))) * (1 / sqrt(2))
    F = isometry(fuse(pspace,pspace), pspace ⊗ pspace')

    @tensor state[-1; -2 -3 -4 -5] := twist(state0,2)[1 2; -2 -3 -4 -5] * F[-1; 1 2]

    @tensor twosite_final[-1 -2; -3 -4] := twosite_op[1 4; 2 5] * twist(F,3)[-1; 1 3] * twist(F,3)[-2; 4 6] * conj(F[-3; 2 3]) * conj(F[-4; 5 6])

    (Nr, Nc) = (2, 2)
    lattice = InfiniteSquare(Nr, Nc)
    pspaces_fused = fill(pspace_fused, Nr, Nc)
    H = PEPSKit.LocalOperator(
        pspaces_fused,
        (neighbor => twosite_final for neighbor in PEPSKit.nearest_neighbours(lattice))...,
    )
            
    wpeps = InfiniteWeightPEPS(InfinitePEPS(state; unitcell = (Nr, Nc)))
    tol = 0.0
    maxiter = floor(0.01 / dt)
    iterations = 90
    βs = [dt*maxiter*i for i = 1:iterations]
    trscheme_peps = truncdim(Dcut) & truncerr(1e-10)
    convert_to_pepo = A -> convert_to_pepo_fuser(A, F)

    occupancies = []
    corrlengths = []
    for iter = 1:iterations
        println("Iteration: $iter")
        alg = SimpleUpdate(dt, tol, maxiter, trscheme_peps)
        result = simpleupdate(wpeps, H, alg; bipartite=false)
        wpeps = result[1]

        peps = InfinitePEPS(wpeps)
        pepo = InfinitePEPO(reshape(convert_to_pepo.(PEPSKit.unitcell(peps)), Nr, Nc, 1))

        vumps_alg = VUMPS(; maxiter = 500)
        E_num, E_ξ = get_both_observables(pepo, envspace, vumps_alg, number_operator)
        push!(occupancies, E_num)
        push!(corrlengths, E_ξ)
    end
    file = jldopen("SF_SU_Dcut_$(Dcut)_χenv_$(χenv)_dt_$(dt).jld2", "w")
    file["βs"] = βs
    file["ns"] = occupancies
    file["ξs"] = corrlengths
    return βs, occupancies, corrlengths

end

Dcuts = [3]
χenvs = [12]
dts = [1e-2]
for Dcut = Dcuts
    for χenv = χenvs
        for dt = dts
            get_data_SU(Dcut, χenv, dt)
        end
    end
end