# function apply_solver(G, A, ::Val{false})
#     # Change name
#     @tensor Ax[-1 -2 -3 -4; -5 -6 -7 -8] := A[-1 -5; 1 2] * G[2 1; -2 -3 -4 -6 -7 -8]
#     return Ax
# end

# function apply_solver(x1, y, ::Val{true})
#     # Change name
#     @tensor Anew[-1 -2; -3 -4] := y[-1 1 2 3; -2 4 5 6] * conj(x1[-3 -4; 1 2 3 4 5 6])
#     return Anew
# end

function apply_A_loop(A, x::TensorMap, ::Val{false})
    Ax = ncon([A, x], [[1 2 -2 -3 -4 -6 -7 -8], [-1 -5 2 1]])
    return permute(Ax, ((1,2,3,4), (5,6,7,8)))
end

function apply_A_loop(A, Ax::TensorMap, ::Val{true})
    x = ncon([A, Ax], [[-4 -3 1 2 3 4 5 6], [-1 1 2 3 -2 4 5 6]], [true false])
    return permute(x, ((1,2),(3,4)))
end

function solve_4_loop_distinct(RHS, space, levels_to_update; ϵ = 1e-10, verbosity = 1, maxiter = 100)
    T = scalartype(RHS)
    pspace = ℂ^2   
    A = randn(T, pspace ⊗ pspace', space ⊗ space')
    As = fill(A, 4)

    for i = 1:maxiter
        G = ncon(As[2:end],  [[-3 -6 -1 1], [-4 -7 1 2], [-5 -8 2 -2]])
        # apply_A = (x, val) -> apply_solver(G, x, val)
        apply_A = (x, val) -> apply_A_loop(G, x, val)
        x, _ = lssolve(apply_A, RHS, LSMR(verbosity = verbosity-1, maxiter = 1000))
        As[1] = copy(x)
        circshift!(As, -1)
        RHS_approx = permute(ncon(As, [[-1 -5 4 1], [-2 -6 1 2], [-3 -7 2 3], [-4 -8 3 4]]), ((1,2,3,4), (5,6,7,8)))
        error = norm(RHS - RHS_approx)/norm(RHS)
        error_dif = maximum([norm(As[i] - As[1]) for i = 2:4])
        println("Error at step $(i): $error. Max dif in As: $error_dif")
        if error_dif < ϵ
            return As
        end
    end
    return As
end