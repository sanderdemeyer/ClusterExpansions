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

function ClusterExpansion(twosite_op, onesite_op; nn_term = nothing, p = 3, verbosity = 0, T = Complex{BigFloat}, spaces = i -> (i >= 0) ? ℂ^(2^(i)) : ℂ^10, symmetry = nothing, solving_loops = true, envspace = χ -> ℂ^χ)
    return ClusterExpansion(twosite_op, onesite_op, nn_term, p, verbosity, T, spaces, symmetry, solving_loops, envspace)
end

function bond_dimension(ce_alg::ClusterExpansion)
    return sum([dim(ce_alg.spaces(i)) for i = 0:div(ce_alg.p,2)]) + (ce_alg.p >= 4) * dim(ce_alg.spaces(-1)) + + (ce_alg.p >= 6) * dim(ce_alg.spaces(-2))
end

function ising_operators(J, g, z; T = Complex{BigFloat}, loop_space = ℂ^10, kwargs...)
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

function ising_operators_Z2(J; T = Complex{BigFloat}, loop_space = ℂ^10, kwargs...)
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

function spinless_fermion_operators(t, V, μ; b = 0.0, δ = 0.0, T = Complex{BigFloat}, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    pspace = Vect[fℤ₂](0 => 1, 1 => 1)

    kinetic_operator = FermionOperators.f_hop(T)
    number_operator = FermionOperators.f_num(T)
    number_operator_halffilling = number_operator - id(pspace)/2
    symmetry_breaking_term = FermionOperators.f⁻f⁻(T) - FermionOperators.f⁺f⁺(T)
    @tensor number_twosite[-1 -2; -3 -4] := number_operator_halffilling[-1; -3] * number_operator_halffilling[-2; -4]
    
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
    return ClusterExpansion(twosite_op, onesite_op; T, spaces, envspace, kwargs...)
end

function spinless_fermion_operators(; kwargs...)
    return spinless_fermion_operators(1.0, 0.0, 0.0; kwargs...)
end

function heisenberg_operators(Jx, Jy, Jz, h; spin = 1//2, spin_symmetry = Trivial, T = Complex{BigFloat}, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    if spin_symmetry == Trivial
        twosite_op =  rmul!(SpinOperators.S_x_S_x(T, spin_symmetry; spin=spin), Jx) +
                rmul!(SpinOperators.S_y_S_y(T, spin_symmetry; spin=spin), Jy) +
                rmul!(SpinOperators.S_z_S_z(T, spin_symmetry; spin=spin), Jz)
        onesite_op = rmul!(SpinOperators.S_z(T, spin_symmetry; spin=spin), h)
    else
        @assert (h == 0) && (Jx == Jy == Jz) "Invalid parameters for given symmetry"
        twosite_op =  rmul!(SpinOperators.S_exchange(T, spin_symmetry; spin=spin), Jx)
        onesite_op = rmul!(id(SpinOperators.spin_space(spin_symmetry; spin=spin)), T(0))
    end

    if spin_symmetry == Trivial
        if sum(iszero.([Jx Jy Jz])) >= 2
            spaces = i -> (i >= 0) ? ℂ^(2^(i)) : loop_space
        else
            spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : loop_space
        end
        envspace = χ -> ℂ^χ
    elseif spin_symmetry == U1Irrep
        loop_space = Vect[U1Irrep](1 => 4, -1 => 4)
        spaces = i -> if i == 0
            Vect[U1Irrep](0 => 1)
        elseif i > 0
            Vect[U1Irrep](1 => 2^(2*i-2), -1 => 2^(2*i-2))
        else
            loop_space
        end
        envspace = χ -> Vect[U1Irrep](0 => div(χ,2), 1 => div(χ,4), -1 => div(χ,4))
    end
    return ClusterExpansion(twosite_op, onesite_op; spaces, envspace, kwargs...)
end

function heisenberg_operators(; kwargs...)
    return heisenberg_operators(1.0, 1.0, 1.0, 0.0; kwargs...)
end

function J1J2_operators(J1, J2, h; spin = 1//2, spin_symmetry = Trivial, T = Complex{BigFloat}, loop_space = Vect[fℤ₂](0 => 5, 1 => 5), kwargs...)
    twosite_op =  rmul!(SpinOperators.S_exchange(T, spin_symmetry; spin), J1)
    nn_term = rmul!(SpinOperators.S_exchange(T, spin_symmetry; spin), J2)
    onesite_op = rmul!(SpinOperators.S_z(T, spin_symmetry; spin), h)

    if spin_symmetry == Trivial
        spaces = i -> (i >= 0) ? ℂ^(2^(2*i)) : loop_space
        envspace = χ -> ℂ^χ
    elseif spin_symmetry == U1Irrep
        loop_space = Vect[U1Irrep](1 => 4, -1 => 4)
        spaces = i -> if i == 0
            Vect[U1Irrep](0 => 1)
        elseif i > 0
            Vect[U1Irrep](1 => 2^(2*i-2), -1 => 2^(2*i-2))
        else
            loop_space
        end
        envspace = χ -> Vect[U1Irrep](0 => div(χ,2), 1 => div(χ,4), -1 => div(χ,4))
    end
    return ClusterExpansion(twosite_op, onesite_op; nn_term, spaces, envspace, kwargs...)
end

function tJ_operators(t, J, μ; t′ = 0.0, particle_symmetry = Trivial, spin_symmetry = Trivial, slave_fermion = false, T = Complex{BigFloat}, filling = 1, kwargs...)
    hopping_operator = TJOperators.e_hop(T, particle_symmetry, spin_symmetry; slave_fermion)
    number_operator = TJOperators.e_num(T, particle_symmetry, spin_symmetry; slave_fermion)
    heisenberg_operators = TJOperators.S_exchange(particle_symmetry, spin_symmetry; slave_fermion) - (filling^2 / 4) * (number_operator ⊗ number_operator)

    twosite_op = rmul!(hopping_operator, -T(t)) + rmul!(heisenberg_operators, T(J))
    onesite_op = rmul!(number_operator, -T(μ))

    if t == 0.0 && J == 0.0
        @warn "Not known when we can use smaller spaces"
    end
    if particle_symmetry == U1Irrep
        spaces = i -> if i == 0
            Vect[fℤ₂ ⊠ U1Irrep]((0,0) => 1)
        elseif i == 1
            Vect[fℤ₂ ⊠ U1Irrep]((0,0) => 5, (1,1) => 2, (1,-1) => 2)
        elseif i > 0
            Vect[fℤ₂ ⊠ U1Irrep]((0,1) => 3, (0,0) => 1, (0,2) => 1, (1,0) => 2, (1,2) => 2)
        else
            Vect[fℤ₂ ⊠ U1Irrep]((0,1) => 20, (1,1) => 10, (1,-1) => 10)
        end
        envspace = χ -> Vect[fℤ₂ ⊠ U1Irrep]((0,1) => div(χ,2), (1,1) => div(χ,4), (1,-1) => div(χ,4))
    else
        spaces = i -> if i == 0
        Vect[fℤ₂](0 => 1)
        elseif i == 1
            Vect[fℤ₂](0 => 5, 1 => 4)
        elseif i > 0
            Vect[fℤ₂](0 => 2*3^(2*i-1), 1 => 2*3^(2*i-1))
        else
            Vect[fℤ₂](0 => 20, 1 => 20)
        end
        envspace = χ -> Vect[fℤ₂](0 => div(χ,2), 1 => div(χ,2))
    end

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

function hubbard_operators(t, U, μ; particle_symmetry = Trivial, spin_symmetry = Trivial, T = Complex{BigFloat}, loop_space = Vect[fℤ₂](0 => 50, 1 => 50), kwargs...)
    pspace = HubbardOperators.hubbard_space(particle_symmetry, spin_symmetry)
    hopping_operator = HubbardOperators.e_hop(T, particle_symmetry, spin_symmetry)
    U_operator = (HubbardOperators.nꜛ(T, particle_symmetry, spin_symmetry) - id(pspace) / 2) *
                 (HubbardOperators.nꜜ(T, particle_symmetry, spin_symmetry) - id(pspace) / 2)
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
