function truncate_hor(Z::TensorMap, space::ElementarySpace)
    @tensor ZZ[-1 -2 -3; -4 -5 -6] := Z[-1 -2; -3 1] * Z[1 -4; -5 -6]
    U, S, V = tsvd(ZZ, trunc = truncspace(space))
    ZL = permute(U * sqrt(S), ((1,2),(3,4)))
    ZR = permute(sqrt(S) * V, ((1,2),(3,4)))

    @tensor ZZ2[-1 -2 -3; -4 -5 -6] := ZR[-1 -2; -3 1] * ZL[1 -4; -5 -6]
    U, S, V = tsvd(ZZ2, trunc = truncspace(space))
    ZL = permute(U * sqrt(S), ((1,2),(3,4)))
    ZR = permute(sqrt(S) * V, ((1,2),(3,4)))

    @assert norm(ZL - ZR)/norm(ZL) < 1e-5
    return (ZL+ZR)/2
end

function truncate_ver(Z::TensorMap, space::ElementarySpace)
    @tensor ZZ[-1 -2 -3; -4 -5 -6] := Z[-1 -2; 1 -3] * Z[-4 1; -5 -6]
    U, S, V = tsvd(ZZ, trunc = truncspace(space))
    ZB = permute(U * sqrt(S), ((1,2),(4,3)))
    ZT = permute(sqrt(S) * V, ((2,1),(3,4)))

    @tensor ZZ2[-1 -2 -3; -4 -5 -6] := ZT[-1 -2; 1 -3] * ZB[-4 1; -5 -6]
    U, S, V = tsvd(ZZ2, trunc = truncspace(space))
    ZB = permute(U * sqrt(S), ((1,2),(4,3)))
    ZT = permute(sqrt(S) * V, ((2,1),(3,4)))

    @assert norm(ZB - ZT)/norm(ZB) < 1e-5
    return (ZB+ZT)/2
end

function truncate_hor(Z::TensorMap, D::Int)
    return truncate_hor(Z, ℂ^D)
end

function truncate_ver(Z::TensorMap, D::Int)
    return truncate_ver(Z, ℂ^D)
end

function truncate_tensor(Z::TensorMap, space::ElementarySpace)
    return truncate_hor(truncate_ver(Z, space), space)
end

function truncate_tensor(Z::TensorMap, D::Int)
    return truncate_hor(truncate_ver(Z, D), D)
end