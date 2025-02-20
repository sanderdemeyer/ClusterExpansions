using TensorKit
using KrylovKit
using Test
using Random

# using LinearAlgebra: exp! as la_exp
# using LinearAlgebra
using GenericLinearAlgebra: svd as generic_svd
using GenericLinearAlgebra: eigen as generic_eigen
using GenericLinearAlgebra: Diagonal
using GenericLinearAlgebra

N = 10
D = 8
T = Complex{BigFloat}
M = convert(Array,rand(T, N,N))
M = (M + M')/2
MB = convert(Array{ComplexF64}, M)

eigval, eigvec = generic_eigen(M; sortby = x -> -abs(x))
eigvalB, eigvecB = generic_eigen(MB; sortby = x -> -abs(x))

@assert (norm(M - eigvec * Diagonal(eigval) * eigvec') < 1e-36)

eigvec_trunc = eigvec[:,1:D]
eigval_trunc = eigval[1:D]
eigvec_truncB = eigvecB[:,1:D]
eigval_truncB = eigvalB[1:D]

println(eigval)
println(eigvalB)
println(sum(abs.(eigval-eigvalB))/N)

println("Truncation error = $(norm(M - eigvec_trunc * Diagonal(eigval_trunc) * eigvec_trunc'))")
println("Truncation errorB = $(norm(MB - eigvec_truncB * Diagonal(eigval_truncB) * eigvec_truncB'))")

println(a)

setprecision(128)

TB = randn(Complex{BigFloat}, ℂ^8, ℂ^8)
TB = (TB + TB') / 2
TA = TensorMap(convert(Vector{ComplexF64},TB.data), ℂ^8, ℂ^8)


function sortfunc(λ)
    println("called")
    return -abs(λ)
end

eigvalA, eigvecA = eig(TA)
eigvalB, eigvecB = eig(TB)

println("Difference between eigenvalues = $(norm(eigvalA-eigvalB))")
println("Difference between eigenvectors = $(norm(eigvecA-eigvecB))")

# println(vals)

println([eigvalB[i,i] for i = 1:8])
println([eigvalA[i,i] for i = 1:8])

TB = randn(Complex{BigFloat}, ℂ^2 ⊗ ℂ^3, ℂ^2 ⊗ ℂ^3)
TB = (TB + TB') / 2
TA = TensorMap(convert(Vector{ComplexF64},TB.data), ℂ^2 ⊗ ℂ^3, ℂ^2 ⊗ ℂ^3)

eigvalA2, eigvecA2 = eig(TA)
eigvalB2, eigvecB2 = eig(TB)

# println([eigvalB2[i,i] for i = 1:6])

println("Difference between eigenvalues = $(norm(eigvalA2-eigvalB2))")
println("Difference between eigenvectors = $(norm(eigvecA2-eigvecB2))")


