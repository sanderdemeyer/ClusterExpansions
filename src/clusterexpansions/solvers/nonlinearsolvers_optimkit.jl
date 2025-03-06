"""
    This is not yet updated to deal with BigFloat
"""

function contract_tensors_symmetric_fused(A)
    loop = ncon([A, A, A, A], [[1 -1 2], [2 -2 3], [3 -3 4], [4 -4 1]])
    return permute(loop, ((1,2),(3,4)))
end

function get_gradient(A)
    return ncon([A, A, A], [[-3 -6 -2 1], [-4 -7 1 2], [-5 -8 2 -1]])
end

function get_gradient_fused(A)
    return ncon([A, A, A], [[-2 -3 1], [1 -4 2], [2 -5 -1]])
end

function my_add!(Y, X, a)
    Y .+= X .* a
    return InfiniteMPS([Y]).AL[1]
end

function get_A_nudge(A, exp_H; c = 0)
    D = contract_tensors_symmetric(A) - exp_H
    g = get_gradient(A)

    x1 = ncon([D, g], [[-1 1 2 3 -2 4 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x2 = ncon([D, g], [[1 -1 2 3 4 -2 5 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x3 = ncon([D, g], [[1 2 -1 3 4 5 -2 6], [-3 -4 1 2 3 4 5 6]], [false true])
    x4 = ncon([D, g], [[1 2 3 -1 4 5 6 -2], [-3 -4 1 2 3 4 5 6]], [false true])

    f = (norm(D)/norm(exp_H))^2 + c*norm(A)^2
    ∂f = permute(x1+x2+x3+x4, ((1,2),(3,4))) + c*A
    return f, ∂f
end

function get_A_nudge_leftgauge(A, exp_H; c = 0)
    D = contract_tensors_symmetric_fused(A) - exp_H
    g = get_gradient_fused(A)
    x1 = ncon([D, g], [[-2 1 2 3], [-1 -3 1 2 3]], [false true])

    f = norm(D)^2 + c*norm(A)^2
    grad = permute(x1, ((1,2),(3,))) + c*A

    V = leftnull(A; alg = QRpos())
    @tensor ∂f[-1 -2; -3] := grad[1 2; -3] * conj(V[1 2; 3]) * V[-1 -2; 3]

    @tensor check[-1; -2] := A[1 2; -1] * conj(∂f[1 2; -2])
    @assert norm(check)/(norm(A)*norm(∂f)) < 1e-14 "Gradient is not in the nullspace of A"

    return f, ∂f
end

# unpack the gradient
function fg_base(ℒ, ψ)
    f, g = Zygote.withgradient(ℒ, ψ)
    return f, g[1]
end

function solve_4_loop_optim(RHS, vspace, levels_to_update; verbosity = 1, symmetry = nothing, x0 = nothing, gradtol = 1e-9)
    # RHS = RHS / norm(RHS)
    D = dim(vspace)
    T = scalartype(RHS)
    
    pspace = ℂ^2
    trivspace = ℂ^1
    if isnothing(x0)
        # A = randn(T, vspace ⊗ fuse(pspace, pspace'), vspace)
        A = randn(T, pspace ⊗ pspace', vspace ⊗ vspace')
        # A = InfiniteMPS([A]).AL[1]
    else
        A = x0
    end
    # A = A / norm(A) * (norm(exp_H))^(1/4) * sqrt(10)

    c = 0

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    # ℒ = A -> norm(contract_tensors_symmetric(A) - exp_H)
    # fg = ψ -> fg_base(ℒ, ψ)
    cfun = x -> get_A_nudge(x, RHS; c = c)

    A, fx, gx, numfg, normgradhistory = optimize(cfun, A, LBFGS(; verbosity=verbosity, gradtol = gradtol); inner=my_inner);

    # Anew = zeros(T, vspace ⊗ pspace ⊗ pspace', vspace)
    # Anew[] = reshape(A[], (D,2,2,D))
    # A = permute(Anew, ((2,3),(1,4)))
    error = norm(contract_tensors_symmetric(A)-RHS) / norm(RHS)
    As = construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = symmetry)

    return As, error, nothing, A
end

function solve_4_loop_optim_leftgauge(RHS, vspace, levels_to_update; verbosity = 1, symmetry = nothing, x0 = nothing, gradtol = 1e-9)
    # RHS = RHS / norm(RHS)
    D = dim(vspace)
    T = scalartype(RHS)
    RHS = permute(RHS, ((1,5,2,6),(3,7,4,8)))
    exp_H = zeros(T, fuse(codomain(RHS)[1], codomain(RHS)[2]) ⊗ fuse(codomain(RHS)[3], codomain(RHS)[4]), fuse(domain(RHS)[1], domain(RHS)[2])' ⊗ fuse(domain(RHS)[3], domain(RHS)[4])')
    exp_H[] = reshape(RHS[], (4,4,4,4))

    pspace = ℂ^2
    trivspace = ℂ^1
    A = randn(T, vspace ⊗ fuse(pspace, pspace'), vspace)
    A = InfiniteMPS([A]).AL[1]

    c = 0
    verbosity = 2

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    # ℒ = A -> norm(contract_tensors_symmetric(A) - exp_H)
    # fg = ψ -> fg_base(ℒ, ψ)
    cfun = x -> get_A_nudge_leftgauge(x, exp_H; c = c)

    A, fx, gx, numfg, normgradhistory = optimize(cfun, A, LBFGS(; verbosity=verbosity, gradtol = gradtol); inner=my_inner);

    error = norm(contract_tensors_symmetric_fused(A)-exp_H) / norm(exp_H)
    Anew = zeros(T, vspace ⊗ pspace ⊗ pspace', vspace)
    Anew[] = reshape(A[], (D,2,2,D))
    A = permute(Anew, ((2,3),(1,4)))

    As = construct_PEPO_loop(A, pspace, vspace, trivspace, levels_to_update; symmetry = symmetry)
    return As, error, nothing, A, normgradhistory
end
