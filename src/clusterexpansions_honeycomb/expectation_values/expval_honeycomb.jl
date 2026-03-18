function _contract_edges_L(O::TensorMap{E,S,2,3}, op::TensorMap{E,S,1,1}, scheme::c3vCTM_honeycomb) where {E,S}
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.L[χNL DNW; χNR] *
    scheme.C[χNR χNE] * scheme.C[χNE; χSEL] * scheme.L[χSEL DE; χSER] *
    scheme.C[χSER; χS] * scheme.C[χS; χSWL] * scheme.L[χSWL DSW; χSWR] *
    flip(twist(O, 2), [4 5])[d1 d2; DNW DE DSW] * op[d2; d1]
end

function PEPSKit.expectation_value(O::TensorMap{E,S,2,3}, op::TensorMap{E,S,1,1}, scheme::c3vCTM_honeycomb) where {E,S}
    return _contract_edges_L(O, op, scheme) / _contract_edges_L(scheme)
end

function _contract_site_large(O::TensorMap{E,S,2,3}, op::TensorMap{E,S,2,2}, scheme::c3vCTM_honeycomb) where {E,S}
    return @tensor opt = true scheme.C[χSWR; χNW] * scheme.C[χNW; χNL] * scheme.L[χNL DNW; χNR] *
    scheme.C[χNR χNEL] * scheme.R[χNEL DE; χNER] *
    scheme.C[χNER; χSE] * scheme.C[χSE; χSL] * scheme.L[χSL DSE; χSR] *
    scheme.C[χSR; χSWL] * scheme.R[χSWL DW; χSWR] *
    flip(twist(O, 2), [4])[dL1 dL2; DNW DE DC] * flip(twist(O, 2), [4 5])[dR1 dR2; DSE DW DC] *
    op[dL2 dR2; dL1 dR1]
end

function PEPSKit.expectation_value(O::TensorMap{E,S,2,3}, op::TensorMap{E,S,2,2}, scheme::c3vCTM_honeycomb) where {E,S}
    op_one = id(scalartype(op), domain(op))
    return _contract_site_large(O, op, scheme) / _contract_site_large(O, op_one, scheme)
end

function PEPSKit.expectation_value(O::TensorMap{E,S,2,3}, symb::Symbol, scheme::c3vCTM_honeycomb) where {E,S}
    if symb == :spectrum
        @tensor opt=true TM[χoutt Dout χoutb; χint Din χinb] := scheme.L[χoutt DNWt; χNt] * scheme.R[χNt DNEt; χint] * 
        scheme.T[DNWt DC DSW] * flip(scheme.T, [2 3])[DNEt DC DSE] * 
        flip(scheme.T, [1 2])[DSW Dout DNWb] * scheme.T[DSE Din; DNEb] * 
        scheme.L[χNb DNWb; χoutb] * scheme.R[χinb DNEb; χNb]

        f = x -> TM * x
        x₀ = randn(scalartype(TM), domain(TM))
        vals, vecs, info = eigsolve(f, x₀, 10, :LM, Arnoldi())
        vals ./= vals[1]
        
        ϵ, δ, θ = marek_gap(vals)
        return 1 / ϵ, δ, θ
    else
        @warn "Observable $(symb) not defined. This will be set to zero"
        return 0
    end
end