function get_marek_gap(O::TensorMap{E,S,2,4}, virtualspace, vumps_alg) where {E,S}
    return get_marek_gap(InfinitePEPO(O), virtualspace, vumps_alg)
end

function get_marek_gap(O::InfinitePEPO, virtualspace, vumps_alg)
    pf = PEPSKit.trace_out(O)
    T = InfiniteMPO([pf[1,1]])

    pspace = domain(pf[1,1])[2]

    mps = InfiniteMPS([
        randn(
            ComplexF64,
            virtualspace * pspace,
            virtualspace,
        )])
    mps, _, _ = leading_boundary(mps, T, vumps_alg);
    ϵ, δ, = marek_gap(mps; num_vals = 20)
    return 1 / ϵ, δ
end

function twosite_vumps(O, virtualspace, vumps_alg, M)
    pf_asym = PEPSKit.trace_out(InfinitePEPO(O))
    T = InfiniteMPO([pf_asym[1,1]])    

    pspace = domain(pf_asym[1,1])[2]
    
    mps = InfiniteMPS([
        randn(
            ComplexF64,
            virtualspace * pspace,
            virtualspace,
        )])
    mps, env, ϵ = leading_boundary(mps, T, vumps_alg);

    E_num = PEPSKit.@autoopt @tensor twist(O,2)[dL1 dL2; DLN Dc DLS DLW] * twist(O,2)[dR1 dR2; DRN DRE DRS Dc] * mps.AC[1][DtL DLN; Dt] * 
    conj(mps.AC[1][DbL DLS; Db]) * mps.AR[2][Dt DRN; DtR] * conj(mps.AR[2][Db DRS; DbR]) * M[dL2 dR2; dL1 dR1] * 
    env.GLs[1][DbL DLW; DtL] * env.GRs[1][DtR DRE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(O,2)[dL dL; DLN Dc DLS DLW] * twist(O,2)[dR dR; DRN DRE DRS Dc] * mps.AC[1][DtL DLN; Dt] * 
    conj(mps.AC[1][DbL DLS; Db]) * mps.AR[2][Dt DRN; DtR] * conj(mps.AR[2][Db DRS; DbR]) *
    env.GLs[1][DbL DLW; DtL] * env.GRs[1][DtR DRE; DbR]
    return E_num / E_denom
end

function onesite_vumps(O, virtualspace, vumps_alg, M)
    pf_asym = PEPSKit.trace_out(InfinitePEPO(O))
    T = InfiniteMPO([pf_asym[1,1]])    

    pspace = domain(pf_asym[1,1])[2]
    
    mps = InfiniteMPS([
        randn(
            ComplexF64,
            virtualspace * pspace,
            virtualspace,
        )])
    mps, env, ϵ = leading_boundary(mps, T, vumps_alg);

    E_num = PEPSKit.@autoopt @tensor twist(O,2)[d1 d2; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) * M[d2; d1] * 
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    E_denom = PEPSKit.@autoopt @tensor twist(O,2)[d d; DN DE DS DW] * mps.AC[1][DtL DN; DtR] * 
    conj(mps.AC[1][DbL DS; DbR]) *
    env.GLs[1][DbL DW; DtL] * env.GRs[1][DtR DE; DbR]
    return E_num / E_denom
end
