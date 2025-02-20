using TensorKit
using KrylovKit
using MPSKitModels
using Graphs
using ClusterExpansions
using OptimKit
using PEPSKit
using Test
using Random
import PEPSKit: rmul!, σᶻᶻ, σˣ, InfiniteSquare

using LinearAlgebra: exp! as la_exp
using LinearAlgebra
using GenericLinearAlgebra: svd as generic_svd
using GenericLinearAlgebra: eigen as generic_eigen
using GenericLinearAlgebra: Diagonal
using GenericLinearAlgebra

# This does not work with BigFloats :'(

setprecision(128)


function fun(x, TA)
    return TA * x
end


T = Complex{Float64}

TB = randn(Complex{Float64}, ℂ^8, ℂ^8)
TB = (TB + TB') / 2
TA = TensorMap(convert(Vector{ComplexF64},TB.data), ℂ^8, ℂ^8)

x0 = randn(ComplexF64, ℂ^8, ℂ^1)

howmany = 1
which = :LM
n = 8
alg = Lanczos(; krylovdim=2 * n, maxiter=10, tol=1e-20)
vals, vecs, info = eigsolve(x -> fun(x,TB), x0, howmany, which, alg)