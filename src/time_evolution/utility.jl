# Fidelity functions

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
    O_stack = repeat(C.A, 1, 1, 2)
    O_stack[:, :, 2] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(C.A)))
    double_layer = InfinitePEPO(O_stack)

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

    network_trunc = InfiniteSquareNetwork(double_layer)
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
    overlap_layer = InfinitePEPO(O_stack)

    O_stack = repeat(A.A, 1, 1, 2)
    O_stack[:, :, 2] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(A.A)))
    top_layer = InfinitePEPO(O_stack)

    O_stack = repeat(B.A, 1, 1, 2)
    O_stack[:, :, 2] .= PEPSKit.unitcell(InfinitePEPO(PEPSKit._dag.(B.A)))
    bot_layer = InfinitePEPO(O_stack)

    network_top = InfiniteSquareNetwork(top_layer)
    env_top, = leading_boundary(CTMRGEnv(network_top, envspace), network_top, ctm_alg);
    norm_top = network_value(network_top, env_top)

    network_bot = InfiniteSquareNetwork(bot_layer)
    env_bot, = leading_boundary(CTMRGEnv(network_bot, envspace), network_bot, ctm_alg);
    norm_bot = network_value(network_bot, env_bot)

    network_overlap = InfiniteSquareNetwork(overlap_layer)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)
    overlap = network_value(network_overlap, env_overlap)

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

# Utility functions for local and exact truncation

function apply_isometry(A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}, Ws::Vector{<:AbstractTensorMap{E,S}}) where {E,S}
    @tensor A_new[-1 -2; -3 -4 -5 -6] := A[1][1 -2; 2 4 6 8] * A[2][-1 1; 3 5 7 9] * Ws[1][2 3; -3] * Ws[2][4 5; -4] * Ws[3][-5; 6 7] * Ws[4][-6; 8 9]
    return A_new
end

function apply_isometry(A::Tuple{AbstractTensorMap{E,S,2,4},AbstractTensorMap{E,S,2,4}}, Ws::Vector{<:AbstractTensorMap{E,S}}, inds::Vector{Int}) where {E,S}
    A_trunc = ncon([A[1], A[2], [Ws[i] for i = setdiff(1:4, inds)]...], [[1, -2, [i ∈ inds ? -i-2 : 2*i for i = 1:4]...], [-1, 1, [i ∈ inds ? -i-6 : 2*i+1 for i = 1:4]...], [dir > 2 ? [-dir-2, 2*dir, 2*dir+1] : [2*dir, 2*dir+1, -dir-2] for dir = setdiff(1:4,inds)]...], fill(false,6-length(inds)))
    return permute(A_trunc, ((1,2),Tuple(3:6+length(inds))))
end

function get_initial_isometry(T, orig_space::ProductSpace, trunc_space::ElementarySpace)
    return [dir > 2 ? isometry(T, orig_space, trunc_space)' : isometry(T, orig_space, trunc_space) for dir = 1:4]
end

# Calculate a PEPO-PEPS exactly.
function apply_PEPO_exact(
    ψ::Union{AbstractTensorMap{E,S,1,4},AbstractTensorMap{E,S,2,4}},
    O::AbstractTensorMap{E,S,2,4},
) where {E,S}
    T = scalartype(ψ)
    orig_space = domain(ψ)[1] ⊗ domain(O)[1]
    trunc_space = fuse(domain(ψ)[1] ⊗ domain(O)[1])
    Ws = get_initial_isometry(T, orig_space, trunc_space)
    return apply_isometry(ψ, O, Ws)
end
