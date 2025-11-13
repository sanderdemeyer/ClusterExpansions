function expectation_value_triangular(A::TensorMap{E,S,2,6}, op::TensorMap{E,S,1,1}, χenv) where {E,S<:ElementarySpace}
    @tensor T[-6 -5 -4; -1 -2 -3] := twist(A, 2)[1 1; -1 -2 -3 -4 -5 -6]
    @tensor Top[-6 -5 -4; -1 -2 -3] := twist(A, 2)[1 2; -1 -2 -3 -4 -5 -6] * op[2; 1]
    
    scheme = c6vCTM_triangular(T)
    scheme = run!(scheme, truncdim(χenv), i -> i > 100)

    return _contract_corners_expval(Top, scheme.C) / _contract_corners_expval(T, scheme.C)
end

function _contract_corners_expval(T, C)
    return @tensor opt = true C[χNW D120; χN] * C[χN D60; χNE] * C[χNE D0; χSE] *
        C[χSE D300; χS] * C[χS D240; χSW] * C[χSW D180; χNW] *
        T[D120 D60 D0; D300 D240 D180]
end
