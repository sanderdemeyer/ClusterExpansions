function contract_tensors_symmetric(A)
    loop = ncon([A, A, A, A], [[1 -1 2], [2 -2 3], [3 -3 4], [4 -4 1]])
    return permute(loop, ((1,2),(3,4)))
end

function get_gradient(A)
    return ncon([A, A, A], [[-2 -3 1], [1 -4 2], [2 -5 -1]])
end

function my_add!(Y, X, a)
    Y .+= X .* a
    return InfiniteMPS([Y]).AL[1]
end

function get_A_nudge(A, exp_H; c = 0)
    D = contract_tensors_symmetric(A) - exp_H
    g = get_gradient(A)
    x1 = ncon([D, g], [[-2 1 2 3], [-1 -3 1 2 3]], [false true])

    f = norm(D)^2 + c*norm(A)^2
    grad = permute(x1, ((1,2),(3,))) + c*A

    V = leftnull(A; alg = QRpos())
    @tensor ∂f[-1 -2; -3] := grad[1 2; -3] * conj(V[1 2; 3]) * V[-1 -2; 3]
    
    @tensor check[-1; -2] := A[1 2; -1] * conj(∂f[1 2; -2])
    @assert norm(check)/(norm(A)*norm(∂f)) < 1e-14 "Gradient is not in the nullspace of A"

    return f, ∂f
    # return norm(D)^2 + c*norm(A)^2, permute(x1+x2+x3+x4, ((1,2),(3,4))) + c*A
end

# unpack the gradient
function fg_base(ℒ, ψ)
    f, g = Zygote.withgradient(ℒ, ψ)
    return f, g[1]
end

function solve_4_loop(RHS, space, levels_to_update; verbosity = 1)
    RHS = RHS / norm(RHS)
    RHS = permute(RHS, ((1,5,2,6),(3,7,4,8)))
    exp_H = TensorMap(zeros, fuse(RHS.codom[1], RHS.codom[2]) ⊗ fuse(RHS.codom[3], RHS.codom[4]), fuse(RHS.dom[1], RHS.dom[2])' ⊗ fuse(RHS.dom[3], RHS.dom[4])')
    println("RHS = $(summary(RHS))")
    println("exp_H = $(summary(exp_H))")
    println("RHS = $(size(reshape(RHS[], (4,4,4,4))))")
    exp_H[] = reshape(RHS[], (4,4,4,4))
    
    pspace = ℂ^2
    trivspace = ℂ^1
    A = TensorMap(randn, space ⊗ fuse(pspace, pspace'), space)
    # A = A / norm(A) * (norm(exp_H))^(1/4) * sqrt(10)
    A = InfiniteMPS([A]).AL[1]

    c = 0
    verbosity = 2

    # Define an inner product to deal with complex numbers
    my_inner(x, y1, y2) = real(dot(y1, y2))

    # ℒ = A -> norm(contract_tensors_symmetric(A) - exp_H)
    # fg = ψ -> fg_base(ℒ, ψ)
    cfun = x -> get_A_nudge(x, exp_H; c = c)

    A, fx, gx, numfg, normgradhistory = optimize(cfun, A, LBFGS(; verbosity=verbosity); inner=my_inner);

    println("norm of A = $(norm(A))")
    println("norm of exp_H = $(norm(exp_H))")
    println("norm of contracted = $(norm(contract_tensors_symmetric(A)))")
    println("norm of error = $(norm(contract_tensors_symmetric(A)-exp_H))")

    println("x = $(summary(x))")
end