function fidelity(A::InfinitePEPS, B::InfinitePEPS, ctm_alg::PEPSKit.CTMRGAlgorithm, envspace::ElementarySpace)
    env_orig, = leading_boundary(CTMRGEnv(A, envspace), A, ctm_alg);
    norm_orig = norm(A, env_orig)

    env_trunc, = leading_boundary(CTMRGEnv(B, envspace), B, ctm_alg);
    norm_trunc = norm(B, env_trunc)

    network = InfiniteSquareNetwork(A, B)
    env_netw, = leading_boundary(CTMRGEnv(network, envspace), network, ctm_alg)
    overlap = network_value(network, env_netw)
    
    return abs(overlap / sqrt(norm_orig*norm_trunc))
end

function fidelity(A::InfinitePEPS, O::InfinitePEPO, B::InfinitePEPS, ctm_alg::PEPSKit.CTMRGAlgorithm, envspace::ElementarySpace)
    O_stack = repeat(O.A, 1, 1, 2)
    O_stack[:, :, 2] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(O.A)))
    OOdag = InfinitePEPO(O_stack)
    
    network_orig = InfiniteSquareNetwork(A, OOdag)
    env_orig, = leading_boundary(CTMRGEnv(network_orig, envspace), network_orig, ctm_alg);
    norm_orig = network_value(network_orig, env_orig)

    env_trunc, = leading_boundary(CTMRGEnv(B, envspace), B, ctm_alg);
    norm_trunc = norm(B, env_trunc)

    network_overlap = InfiniteSquareNetwork(A, O, B)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)
    overlap = network_value(network_overlap, env_overlap)
    
    return abs(overlap / sqrt(norm_orig*norm_trunc))
end

function fidelity(A::InfinitePEPO, B::InfinitePEPO, C::InfinitePEPO, ctm_alg::PEPSKit.CTMRGAlgorithm, envspace::ElementarySpace)
    O_stack = repeat(A.A, 1, 1, 3)
    O_stack[:, :, 2] .= B.A
    O_stack[:, :, 3] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(C.A)))
    triple_layer = InfinitePEPO(O_stack)
    
    O_stack = repeat(A.A, 1, 1, 4)
    O_stack[:, :, 2] .= B.A
    O_stack[:, :, 3] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(B.A)))
    O_stack[:, :, 4] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(A.A)))
    quadruple_layer = InfinitePEPO(O_stack)

    network_orig = InfiniteSquareNetwork(quadruple_layer)
    env_orig, = leading_boundary(CTMRGEnv(network_orig, envspace), network_orig, ctm_alg);
    norm_orig = network_value(network_orig, env_orig)

    network_trunc = InfiniteSquareNetwork(C)
    env_trunc, = leading_boundary(CTMRGEnv(network_trunc, envspace), network_trunc, ctm_alg);
    norm_trunc = network_value(network_trunc, env_trunc)

    network_overlap = InfiniteSquareNetwork(triple_layer)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)
    overlap = network_value(network_overlap, env_overlap)
    
    return abs(overlap / sqrt(norm_orig*norm_trunc))
end

function fidelity(A::InfinitePEPO, B::InfinitePEPO, ctm_alg::PEPSKit.CTMRGAlgorithm, envspace::ElementarySpace)
    O_stack = repeat(A.A, 1, 1, 2)
    O_stack[:, :, 2] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(B.A)))
    double_layer = InfinitePEPO(O_stack)
    network_overlap = InfiniteSquareNetwork(double_layer)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)
    overlap = network_value(network_overlap, env_overlap)

    network_top = InfiniteSquareNetwork(A)
    env_top, = leading_boundary(CTMRGEnv(network_top, envspace), network_top, ctm_alg);
    norm_top = network_value(network_top, env_top)

    network_bot = InfiniteSquareNetwork(A)
    env_bot, = leading_boundary(CTMRGEnv(network_bot, envspace), network_bot, ctm_alg);
    norm_bot = network_value(network_bot, env_bot)
    return abs(overlap / sqrt(norm_top*norm_bot))
end

function fidelity(A::AbstractTensorMap{E,S,1,4}, B::AbstractTensorMap{E,S,1,4}, ctm_alg, envspace) where {E,S}
    return fidelity(InfinitePEPS(A), InfinitePEPS(B), ctm_alg, envspace)
end

function fidelity(A::AbstractTensorMap{E,S,2,4}, B::AbstractTensorMap{E,S,2,4}, ctm_alg, envspace) where {E,S}
    return fidelity(InfinitePEPO(A), InfinitePEPO(B), ctm_alg, envspace)
end

function fidelity(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, B::AbstractTensorMap{E,S,1,4}, ctm_alg, envspace) where {E,S}
    return fidelity(InfinitePEPS(A), InfinitePEPO(O), InfinitePEPS(B), ctm_alg, envspace)
end

function fidelity(A::Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}, B::AbstractTensorMap{E,S,1,4}, ctm_alg, envspace) where {E,S}
    return fidelity(A..., B, ctm_alg, envspace)
end

function fidelity(A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}, B::AbstractTensorMap{E,S,2,4}, ctm_alg, envspace) where {E,S}
    return fidelity(A..., B, ctm_alg, envspace)
end

function get_initial_isometry(T, cod::ElementarySpace, dom::ElementarySpace, func)
    # return [dir > 2 ? func(T, cod', dom') : func(T, cod, dom) for dir = 1:4]
    return [func(T, cod, dom) for dir = 1:4]
end

function get_initial_isometry(T, cod::ProductSpace, dom::ElementarySpace, func)
    # return [dir > 2 ? func(T, prod([i' for i = cod]), dom') : func(T, cod, dom) for dir = 1:4]
    return [func(T, cod, dom) for dir = 1:4]
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, Ws::Vector{<:AbstractTensorMap{E,S,1,1}}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[-1; 1 2 3 4] * Ws[1][1; -2] * Ws[2][2; -3] * conj(Ws[3][3; -4]) * conj(Ws[4][4; -5])
    return A_trunc
end

function apply_isometry_hor(A::AbstractTensorMap{E,S,1,4}, W::AbstractTensorMap{E,S,1,1}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[-1; -2 1 -4 2] * W[1; -3] * conj(W[2; -5])
    return A_trunc
end

function apply_isometry_ver(A::AbstractTensorMap{E,S,1,4}, W::AbstractTensorMap{E,S,1,1}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[-1; 1 -3 2 -5] * W[1; -2] * conj(W[2; -4])
    return A_trunc
end

function apply_isometry(O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,1,1}}) where {E,S}
    @tensor O_trunc[-1 -2; -3 -4 -5 -6] := O[-1 -2; 1 2 3 4] * Ws[1][1; -3] * Ws[2][2; -4] * conj(Ws[3][3; -5]) * conj(Ws[4][4; -6])
    return O_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    @tensor A_trunc[-1; -2 -3 -4 -5] := A[1; 2 4 6 8] * O[-1 1; 3 5 7 9] * Ws[1][2 3; -2] * Ws[2][4 5; -3] * conj(Ws[3][6 7; -4]) * conj(Ws[4][8 9; -5])
    return A_trunc
end

function apply_isometry(O₁::AbstractTensorMap{E,S,2,4}, O₂::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    @tensor O_trunc[-1 -2; -3 -4 -5 -6] := twist(O₁, (5,6))[1 -2; 2 4 6 8] * twist(O₂, (5,6))[-1 1; 3 5 7 9] * Ws[1][2 3; -3] * Ws[2][4 5; -4] * conj(Ws[3][6 7; -5]) * conj(Ws[4][8 9; -6])
    return O_trunc
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, Ws::Vector{<:AbstractTensorMap{E,S,1,1}}, inds::Vector{Int}) where {E,S}
    A_trunc = ncon([A, [Ws[i] for i = setdiff(1:4, inds)]...], [[-1, [i ∈ inds ? -i-1 : i+1 for i = 1:4]...], [[i+1 -i-1] for i = setdiff(1:4,inds)]...], vcat(false,[dir > 2 ? true : false for dir = 1:4 if dir ∉ inds]))
    return permute(A_trunc, ((1,),(2,3,4,5)))
end

function apply_isometry(O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,1,1}}, inds::Vector{Int}) where {E,S}
    O_trunc = ncon([O, [Ws[i] for i = setdiff(1:4, inds)]...], [[-1, -2, [i ∈ inds ? -i-2 : i+2 for i = 1:4]...], [[i+2 -i-2] for i = setdiff(1:4,inds)]...], vcat(false,[dir > 2 ? true : false for dir = 1:4 if dir ∉ inds]))
    return permute(O_trunc, ((1,2),(3,4,5,6)))
end

function apply_isometry(A::AbstractTensorMap{E,S,1,4}, O::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}, inds::Vector{Int}) where {E,S}
    A_trunc = ncon([A, O, [Ws[i] for i = setdiff(1:4, inds)]...], [[1, [i ∈ inds ? -i-1 : 2*i for i = 1:4]...], [-1, 1, [i ∈ inds ? -i-5 : 2*i+1 for i = 1:4]...], [[2*i, 2*i+1, -i-1] for i = setdiff(1:4,inds)]...], vcat(false,false,[dir > 2 ? true : false for dir = 1:4 if dir ∉ inds]))
    return permute(A_trunc, ((1,),Tuple(2:5+length(inds))))
end

function apply_isometry(O₁::AbstractTensorMap{E,S,2,4}, O₂::AbstractTensorMap{E,S,2,4}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}, inds::Vector{Int}) where {E,S}
    A_trunc = ncon([twist(O₁,(5,6)), twist(O₂,(5,6)), [Ws[i] for i = setdiff(1:4, inds)]...], [[1, -2, [i ∈ inds ? -i-2 : 2*i for i = 1:4]...], [-1, 1, [i ∈ inds ? -i-6 : 2*i+1 for i = 1:4]...], [[2*i, 2*i+1, -i-2] for i = setdiff(1:4,inds)]...], vcat(false,false,[dir > 2 ? true : false for dir = 1:4 if dir ∉ inds]))
    return permute(A_trunc, ((1,2),Tuple(3:6+length(inds))))
end

function apply_isometry(A::Union{Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}) where {E,S}
    return apply_isometry(A..., Ws)
end

function apply_isometry(A::Union{Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}}}, Ws::Vector{<:AbstractTensorMap{E,S,2,1}}, inds::Vector{Int}) where {E,S}
    return apply_isometry(A..., Ws, inds)
end

# Utility function to update the isometry. t is the result of `contract_34_patch`.
function update_isometry(t, trscheme)
    T = scalartype(t)
    U, _, V = tsvd(t, trunc = trscheme)
    # Q, R = leftorth(U; alg = QRpos())
    Ws_new = [zeros(T, codomain(U), domain(U)) for dir = 1:4]
    for dir in 1:4
        Ws_new[dir] = copy(U)
    end
    return Ws_new
end

function get_trunc_space(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4}} where {E,S<:ElementarySpace}
)
    return domain(A)[1]
end

function get_trunc_space(
    A::Union{Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace}
)
    return domain(A[1])[1] ⊗ domain(A[2])[1]
end

function get_top_space(
    A::Union{AbstractTensorMap{E,S,1,4}, AbstractTensorMap{E,S,2,4}} where {E,S<:ElementarySpace}
)
    return domain(A)[1]
end

function get_top_space(
    A::Union{Tuple{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}} where {E,S<:ElementarySpace}
)
    return domain(A[1])[1]
end

# Utility functions for NoEnvTruncation

# QR decomposition
function QR_proj(A, p1, p2; check_space=true)
    len = length(codomain(A)) + 8
    q1 = Tuple(setdiff(1:len, p1))
    _, RA1 = leftorth(A, (q1, p1))
    q2 = Tuple(setdiff(1:len, p2))
    RA2, _ = rightorth(A, (p2, q2))
    if check_space
        if domain(RA1) != codomain(RA2)
            throw(SpaceMismatch("domain and codomain of projectors do not match"))
        end
    end
    return RA1, RA2
end

function oblique_projector(R1, R2, trunc)
    mat = R1 * R2
    U, S, Vt = tsvd(mat; trunc)

    P1 = R2 * adjoint(Vt) * inv(sqrt(S))
    P2 = inv(sqrt(S)) * adjoint(U) * R1
    return P1, P2
end

function find_proj(A, p1, p2, trunc; check_space=true)
    R1, R2 = QR_proj(A, p1, p2; check_space=check_space)
    P1, P2 = oblique_projector(R1, R2, trunc)
    return P1, P2
end
