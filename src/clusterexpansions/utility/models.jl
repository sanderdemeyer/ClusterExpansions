struct ClusterExpansion
    twosite_op
    onesite_op
    nn_term
    p
    verbosity
    T
    spaces
    symmetry
    solving_loops
    envspace
end

function ClusterExpansion(twosite_op, onesite_op; nn_term = nothing, p = 3, verbosity = 0, T = ComplexF64, spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10, symmetry = "C4", solving_loops = true, envspace = χ -> ℂ^χ)
    return ClusterExpansion(twosite_op, onesite_op, nn_term ,p, verbosity, T, spaces, symmetry, solving_loops, envspace)
end

function bond_dimension(ce_alg::ClusterExpansion)
    return sum([dim(ce_alg.spaces(i)) for i = 0:div(ce_alg.p,2)]) + (ce_alg.p >= 4) * dim(ce_alg.spaces(-1)) + + (ce_alg.p >= 6) * dim(ce_alg.spaces(-2))
end

function ising_operators(J, g, z; T = ComplexF64, loop_space = ℂ^10, kwargs...)
    twosite_op = rmul!(PEPSKit.σᶻᶻ(T), -J)
    onesite_op = rmul!(PEPSKit.σˣ(T), g * -J) + rmul!(PEPSKit.σᶻ(T), z * -J)

    if g == 0.0
        spaces = i -> (i >= 0) ? ℂ^(2^(i)) : loop_space
    else
        spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : loop_space
    end
    return ClusterExpansion(twosite_op, onesite_op; spaces, kwargs...)
end

function ising_operators(; kwargs...)
    return ising_operators(1.0, 0.0, 0.0; kwargs...)
end

function ising_operators_Z2(J; T = ComplexF64, loop_space = ℂ^10, kwargs...)
    twosite_op = rmul!(PEPSKit.σᶻᶻ(T, Z2Irrep), -J)
    pspace = Z2Space(0 => 1, 1 => 1)
    onesite_op = rmul!(id(pspace), 0.0)

    spaces = i -> if i == 0
        Z2Space(0 => 1)
    elseif i > 0
        Z2Space(0 => 2^(i-1), 1 => 2^(i-1))
    else
        loop_space
    end
    envspace = χ -> Z2Space(0 => div(χ,2), 1 => div(χ,2))
    return ClusterExpansion(twosite_op, onesite_op; spaces, envspace, kwargs...)
end

function ising_operators_Z2(; kwargs...)
    return ising_operators_Z2(1.0; kwargs...)
end

function spinless_fermion_operators(t, V, μ; b = 0.0, δ = 0.0, T = ComplexF64, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    kinetic_operator = FermionOperators.f_hop(T)
    number_operator = FermionOperators.f_num(T)
    symmetry_breaking_term = FermionOperators.f⁻f⁻(T) - FermionOperators.f⁺f⁺(T)
    @tensor number_twosite[-1 -2; -3 -4] := number_operator[-1; -3] * number_operator[-2; -4]
    twosite_op = rmul!(kinetic_operator, -T(t)) + rmul!(number_twosite, T(V)) - rmul!(symmetry_breaking_term, T(δ))
    onesite_op = rmul!(number_operator, -T(μ)) + rmul!(number_operator, 2*T(V)*T(b))

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
    envspace = χ -> Vect[fℤ₂](0 => div(χ,2), 1 => div(χ,2))
    return ClusterExpansion(twosite_op, onesite_op; spaces, envspace, kwargs...)
end

function spinless_fermion_operators(; kwargs...)
    return spinless_fermion_operators(1.0, 0.0, 0.0; kwargs...)
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

function heisenberg_operators(; kwargs...)
    return heisenberg_operators(-1.0, 1.0, -1.0, 0.0; kwargs...)
end

function tJ_operators(t, J, μ; t′ = 0.0, particle_symmetry = Trivial, spin_symmetry = Trivial, slave_fermion = false, T = ComplexF64, loop_space = Vect[fℤ₂](0 => 20, 1 => 20), kwargs...)
    hopping_operator = TJOperators.e_hop(T, particle_symmetry, spin_symmetry; slave_fermion)
    number_operator = TJOperators.e_num(T, particle_symmetry, spin_symmetry; slave_fermion)
    heisenberg_operators = TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) - (1 / 4) * (number_operator ⊗ number_operator)

    twosite_op = rmul!(hopping_operator, -T(t)) + rmul!(heisenberg_operators, T(J))
    onesite_op = rmul!(number_operator, -T(μ))

    if t == 0.0 && J == 0.0
        @warn "Not known when we can use smaller spaces"
    end
    spaces = i -> if i == 0
        Vect[fℤ₂](0 => 1)
    elseif i == 1
        Vect[fℤ₂](0 => 5, 1 => 4)
    elseif i > 0
        Vect[fℤ₂](0 => 2*3^(2*i-1), 1 => 2*3^(2*i-1))
    else
        loop_space
    end
    envspace = χ -> Vect[fℤ₂](0 => div(χ,2), 1 => div(χ,2))

    if t′ == 0.0
        nn_term = nothing
    else
        nn_term = rmul!(hopping_operator, -T(t′))
    end
    return ClusterExpansion(twosite_op, onesite_op; spaces, envspace, nn_term, kwargs...)
end

function tJ_operators(; kwargs...)
    return tJ_operators(2.5, 1.0, 0.0; kwargs...)
end

function hubbard_operators(t, U, μ; particle_symmetry = Trivial, spin_symmetry = Trivial, T = ComplexF64, loop_space = Vect[fℤ₂](0 => 50, 1 => 50), kwargs...)
    hopping_operator = HubbardOperators.e_hop(T, particle_symmetry, spin_symmetry)
    U_operator = HubbardOperators.nꜛꜜ(T, particle_symmetry, spin_symmetry)
    number_operator = HubbardOperators.e_num(T, particle_symmetry, spin_symmetry)
    twosite_op = rmul!(hopping_operator, -T(t))
    onesite_op = rmul!(number_operator, -T(μ)) + rmul!(U_operator, T(U))

    if t == 0.0 && J == 0.0
        @warn "Not known when we can use smaller spaces"
    end
    spaces = i -> if i == 0
        Vect[fℤ₂](0 => 1)
    elseif i > 0
        Vect[fℤ₂](0 => 2*4^(2*i-1), 1 => 2*4^(2*i-1))
    else
        loop_space
    end
    envspace = χ -> Vect[fℤ₂](0 => div(χ,2), 1 => div(χ,2))
    return ClusterExpansion(twosite_op, onesite_op; spaces, envspace, kwargs...)
end

function hubbard_operators(; kwargs...)
    return hubbard_operators(1.0, 0.0, 0.0; kwargs...)
end
