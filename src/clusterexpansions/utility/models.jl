struct ClusterExpansion
    twosite_op
    onesite_op
    p
    verbosity
    T
    spaces
    symmetry
    solving_loops
end

function ClusterExpansion(twosite_op, onesite_op; p = 3, verbosity = 0, T = ComplexF64, spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10, symmetry = "C4", solving_loops = true)
    return ClusterExpansion(twosite_op, onesite_op, p, verbosity, T, spaces, symmetry, solving_loops)
end

function evolution_operator(ce_alg::ClusterExpansion, β::Number)
    _, O_clust_full = clusterexpansion(ce_alg.T, ce_alg.p, β, ce_alg.twosite_op, ce_alg.onesite_op; spaces = ce_alg.spaces, verbosity = ce_alg.verbosity, symmetry = ce_alg.symmetry, solving_loops = ce_alg.solving_loops)
    O_clust_full = convert(TensorMap, O_clust_full)
    O = zeros(ComplexF64, codomain(O_clust_full), domain(O_clust_full))
    for (f_full, f_conv) in zip(blocks(O_clust_full), blocks(O))
        f_conv[2] .= f_full[2]
    end
    return O
end

function ising_operators(J, g; T = ComplexF64, loop_space = ℂ^10, kwargs...)
    twosite_op = rmul!(PEPSKit.σᶻᶻ(T), -J)
    onesite_op = rmul!(PEPSKit.σˣ(T), g * -J)

    if g == 0.0
        spaces = i -> (i >= 0) ? ℂ^(2^(i)) : loop_space
    else
        spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : loop_space
    end
    return ClusterExpansion(twosite_op, onesite_op; spaces, kwargs...)
end

function spinless_fermion_operators(t, V, μ; T = ComplexF64, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    kinetic_operator = -T(t) * FermionOperators.f_hop(T)
    number_operator = FermionOperators.f_num(T)
    @tensor number_twosite[-1 -2; -3 -4] := number_operator[-1; -3] * number_operator[-2; -4]
    twosite_op = rmul!(kinetic_operator, -T(t)) + rmul!(number_twosite, T(V))
    onesite_op = rmul!(number_operator, -T(μ))

    if t == 0.0
        spaces = i -> if i == 0
            Vect[fℤ₂](0 => 1)
        elseif i > 0
            Vect[fℤ₂](0 => 2^(i-1), 1 => 2^(i-1))
        else
            loop_space
        end
    else
        spaces = i -> if i == 0
            Vect[fℤ₂](0 => 1)
        elseif i > 0
            Vect[fℤ₂](0 => 2^(2*i-1), 1 => 2^(2*i-1))
        else
            loop_space
        end
    end
    return ClusterExpansion(twosite_op, onesite_op; spaces, kwargs...)
end

function heisenberg_operators(Jx, Jy, Jz, h; spin = 1//2, T = ComplexF64, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    twosite_op =  rmul!(S_xx(T; spin=spin), Jx) +
            rmul!(S_yy(T; spin=spin), Jy) +
            rmul!(S_zz(T; spin=spin), Jz)
    onesite_op = rmul!(S_z(T; spin=spin), h)

    if sum(iszero.([Jx Jy Jz])) >= 2
        spaces = i -> (i >= 0) ? ℂ^(2^(i)) : loop_space
    else
        spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : loop_space
    end
    return ClusterExpansion(twosite_op, onesite_op; spaces, kwargs...)
end