function _contract_site_double(A::Tuple{T, T}, env::CTMRGEnv) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
    PEPSKit.@autoopt @tensor t[DpL1 DpR1; DpL2 DpR2] := env.corners[1, 1, 1][χ10; χ1] * env.edges[1, 1, 1][χ1 DNL1 DNL2; χ2] *
        env.edges[1, 1, 1][χ2 DNR1 DNR2; χ3] * env.corners[2, 1, 1][χ3; χ4] *
        env.edges[2, 1, 1][χ4 DE1 DE2; χ5] * env.corners[3, 1, 1][χ5; χ6] *
        env.edges[3, 1, 1][χ6 DSR1 DSR2; χ7] * env.edges[3, 1, 1][χ7 DSL1 DSL2; χ8] *
        env.corners[4, 1, 1][χ8; χ9] * env.edges[4, 1, 1][χ9 DW1 DW2; χ10] *
        A[1][DpL DpL2; DNL1 DC1 DSL1 DW1] * A[2][DpL1 DpL; DNL2 DC2 DSL2 DW2] *
        A[1][DpR DpR2; DNR1 DE1 DSR1 DC1] * A[2][DpR1 DpR; DNR2 DE2 DSR2 DC2]
    return t
end

function _contract_site_single(B::T, env::CTMRGEnv) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
    PEPSKit.@autoopt @tensor t[DpL1 DpR1; DpL2 DpR2] := env.corners[1, 1, 1][χ10; χ1] * env.edges[1, 1, 1][χ1 DNL1; χ2] *
        env.edges[1, 1, 1][χ2 DNR1; χ3] * env.corners[2, 1, 1][χ3; χ4] *
        env.edges[2, 1, 1][χ4 DE1; χ5] * env.corners[3, 1, 1][χ5; χ6] *
        env.edges[3, 1, 1][χ6 DSR1; χ7] * env.edges[3, 1, 1][χ7 DSL1; χ8] *
        env.corners[4, 1, 1][χ8; χ9] * env.edges[4, 1, 1][χ9 DW1; χ10] *
        B[DpL1 DpL2; DNL1 DC1 DSL1 DW1] * B[DpR1 DpR2; DNR1 DE1 DSR1 DC1]
    return t
end

function _contract_corners(env::CTMRGEnv)
    C_NW = env.corners[1, 1, 1]
    C_NE = env.corners[2, 1, 1]
    C_SE = env.corners[3, 1, 1]
    C_SW = env.corners[4, 1, 1]
    return @tensor C_NW[1; 2] * C_NE[2; 3] * C_SE[3; 4] * C_SW[4; 1]
end

function _contract_vertical_edges_double(env::CTMRGEnv)
    return PEPSKit.@autoopt @tensor env.corners[1, 1, 1][χ6; χ1] * env.corners[2, 1, 1][χ1; χ2] *
        env.edges[2, 1, 1][χ2 DC1 DC2; χ3] * env.corners[3, 1, 1][χ3; χ4] *
        env.corners[4, 1, 1][χ4; χ5] * env.edges[4, 1, 1][χ5 DC1 DC2; χ6]
end

function _contract_vertical_edges_single(env::CTMRGEnv)
    return PEPSKit.@autoopt @tensor env.corners[1, 1, 1][χ6; χ1] * env.corners[2, 1, 1][χ1; χ2] *
        env.edges[2, 1, 1][χ2 DC; χ3] * env.corners[3, 1, 1][χ3; χ4] *
        env.corners[4, 1, 1][χ4; χ5] * env.edges[4, 1, 1][χ5 DC; χ6]
end

function _contract_horizontal_edges_double(env::CTMRGEnv)
    return PEPSKit.@autoopt @tensor env.corners[1, 1, 1][χ8; χ1] * env.edges[1, 1, 1][χ1 DCL1 DCL2; χ2] *
        env.edges[1, 1, 1][χ2 DCR1 DCR2; χ3] * env.corners[2, 1, 1][χ3; χ4] *
        env.corners[3, 1, 1][χ4; χ5] * env.edges[3, 1, 1][χ5 DCR1 DCR2; χ6] *
        env.edges[3, 1, 1][χ6 DCL1 DCL2; χ7] * env.corners[4, 1, 1][χ7; χ8]
end

function _contract_horizontal_edges_single(env::CTMRGEnv)
    return PEPSKit.@autoopt @tensor env.corners[1, 1, 1][χ8; χ1] * env.edges[1, 1, 1][χ1 DCL; χ2] *
        env.edges[1, 1, 1][χ2 DCR; χ3] * env.corners[2, 1, 1][χ3; χ4] *
        env.corners[3, 1, 1][χ4; χ5] * env.edges[3, 1, 1][χ5 DCR; χ6] *
        env.edges[3, 1, 1][χ6 DCL; χ7] * env.corners[4, 1, 1][χ7; χ8]
end

function network_value_double(A::Tuple{T, T}, env::CTMRGEnv) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
    return _contract_site_double(A, env) * _contract_corners(env) /
        _contract_vertical_edges_double(env) / _contract_horizontal_edges_double(env)
end

function network_value_single(B::T, env::CTMRGEnv) where {E, S, T <: AbstractTensorMap{E, S, 2, 4}}
    return _contract_site_single(B, env) * _contract_corners(env) /
        _contract_vertical_edges_single(env) / _contract_horizontal_edges_single(env)
end
