function check_loop(As, RHS, spaces)
    trivspace = spaces(0)
    conjugated = [2 6 7 8]
    tens = [(c ∈ conjugated) ? Tensor(ones, trivspace') : Tensor(ones, trivspace) for c in 1:8]
    @tensor RHS_check[-1 -2 -3 -4; -5 -6 -7 -8] := As[1][-4 -8; 1 9 12 2] * As[2][-3 -7; 3 4 10 9] * As[3][-2 -6; 10 5 6 11] * As[4][-1 -5; 12 11 7 8] * tens[1][1] * tens[2][2] * tens[3][3] * tens[4][4] * tens[5][5] * tens[6][6] * tens[7][7] * tens[8][8]
    return norm(RHS_check - RHS) / norm(RHS)
end

function construct_PEPO_loop(A_NW)
    A_SW = rotl90_fermionic(A_NW)
    A_SE = rotl90_fermionic(A_SW)
    A_NE = rotl90_fermionic(A_SE)
    return [A_NW, A_NE, A_SE, A_SW]
end

function solve_4_loop_optim(RHS, spaces, levels_to_update; verbosity = 1, symmetry = nothing, gradtol = 1.0e-9)
    T = scalartype(RHS)
    vspace = spaces(-1)
    trivspace = spaces(0)
    pspace = codomain(RHS)[1]

    opt_alg = LBFGS(; maxiter = 500, gradtol = gradtol, verbosity)

    if isnothing(symmetry)
        A_NW = randn(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
        A_NE = randn(T, pspace ⊗ pspace', trivspace ⊗ trivspace ⊗ vspace' ⊗ vspace')
        A_SE = randn(T, pspace ⊗ pspace', vspace ⊗ trivspace ⊗ trivspace' ⊗ vspace')
        A_SW = randn(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ trivspace' ⊗ trivspace')
        A_NW *= norm(RHS)^(1 / 4) / norm(A_NW)
        A_NE *= norm(RHS)^(1 / 4) / norm(A_NE)
        A_SE *= norm(RHS)^(1 / 4) / norm(A_SE)
        A_SW *= norm(RHS)^(1 / 4) / norm(A_SW)
        As = [A_NW, A_NE, A_SE, A_SW]

        custom_costfun = ψ -> check_loop(ψ, RHS, spaces)

        # optimize free energy per site
        As_final, f, = optimize(
            As,
            opt_alg;
            inner = PEPSKit.real_inner,
        ) do psi
            E, gs = withgradient(psi) do ψ
                return custom_costfun(ψ)
            end
            g = only(gs)
            return E, g
        end
    elseif symmetry == "C4"
        A = randn(T, pspace ⊗ pspace', trivspace ⊗ vspace ⊗ vspace' ⊗ trivspace')
        A *= norm(RHS)^(1 / 4) / norm(A)

        custom_costfun = ψ -> check_loop(construct_PEPO_loop(ψ), RHS, spaces)

        # optimize free energy per site
        A_final, f, = optimize(
            A,
            opt_alg;
            inner = PEPSKit.real_inner,
        ) do psi
            E, gs = withgradient(psi) do ψ
                return custom_costfun(ψ)
            end
            g = only(gs)
            return E, g
        end
        As_final = construct_PEPO_loop(A_final)
    else
        @error "Symmetry $(symmetry) not implemented"
    end

    dict = Dict((0, -1, -1, 0) => 1, (0, 0, -1, -1) => 2, (-1, 0, 0, -1) => 3, (-1, -1, 0, 0) => 4)
    values = [dict[key] for key in levels_to_update]
    As = [As_final[values[1]], As_final[values[2]], As_final[values[3]], As_final[values[4]]]
    return As, f
end
