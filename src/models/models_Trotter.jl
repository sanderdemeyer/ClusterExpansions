abstract type TrotterDecomposition end

struct TwositeTrotterDecomposition <: TrotterDecomposition
    twosite_op
    g
    spaces
end

struct GenericTrotterDecomposition <: TrotterDecomposition
    twosite_op
    onesite_op
    g
    spaces
end

function get_Trotter_onesite(op::AbstractTensorMap{E,S,1,1}, g::Number, β::Number) where {E,S}
    pspace = domain(op)[1]
    return cosh(g*β/2) * id(pspace)  + sinh(g*β/2) * op
end

function get_Trotter_twosite(op::AbstractTensorMap{E,S,1,1}, vspace, β::Number) where {E,S}
    T = scalartype(op)
    pspace = domain(op)[1]
    tensor_base = cosh(β)^2 * sqrt(tanh(β)) * op
    UZZ = zeros(T, pspace ⊗ pspace', vspace ⊗ vspace ⊗ vspace' ⊗ vspace')
    for i = 1:2, j = 1:2, k = 1:2, l = 1:2
        UZZ[][:,:,i,j,k,l] = tensor_base[]^(i+j+k+l)
    end
    return UZZ
end

function ising_operators_Trotter(J, g, z; spin_symmetry = Trivial, T = ComplexF64)
    @assert z == 0 "Only z = 0 implemented. Probably easily extendable to nonzero z values"
    twosite_op = rmul!(PEPSKit.σᶻ(T, spin_symmetry), sqrt(J)) # This sqrt might be wrong
    onesite_op = PEPSKit.σˣ(T)
    spaces = i -> i == 0 ? ℂ^1 : ℂ^2
    return GenericTrotterDecomposition(twosite_op, onesite_op, g, spaces)
end

# function ising_operators_Trotter(J, g, z; spin_symmetry = Trivial, T = ComplexF64)
#     @assert z == 0 "Only z = 0 implemented. Probably easily extendable to nonzero z values"
#     twosite_op = rmul!(PEPSKit.σᶻ(T, spin_symmetry), sqrt(J)) # This sqrt might be wrong
#     onesite_op = PEPSKit.σˣ(T)
#     spaces = i -> i == 0 ? ℂ^1 : ℂ^2
#     return TrotterDecomposition(twosite_op, onesite_op, g, spaces)
# end
