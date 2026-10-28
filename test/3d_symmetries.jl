using TensorKit
using ClusterExpansions
using PEPSKit
using TensorKitTensors
using Test
using Random

# model = heisenberg_operators
# param = (-1.0, -1.0, -1.0, 0.0)
model = ising_operators
param = (1.0, 0.0, 0.0)
kw = (spin_symmetry = Trivial,)
model = bose_hubbard_operators
param = (1.0, 1.0, 0.5, -0.5)
kw = ()
p = 3
T = ComplexF64
β₀ = 0.02
Δβ = 0.02
maxiter = 2
Dcut = 5

time_alg = UniformTimeEvolution(β₀, Δβ, maxiter; verbosity = 0)

for (i,symmetry) = enumerate([nothing "Oh"])
    trunc_alg = NoEnvTruncation(truncrank(Dcut); rot_invariance = !isnothing(symmetry))
    ce_alg = model(param...; kw..., p, T, symmetry)
    O = time_evolve_cubic(ce_alg, time_alg, trunc_alg, (O,i) -> []; normalizing = true)[3][end]

    check_rot_ud = norm(flip(permute(O, ((1,2),(3,5,7,6,8,4))), [5 8]) - O) < 1e-9
    check_rot_ns = norm(flip(permute(O, ((1,2),(5,4,6,8,7,3))), [5 8]) - O) < 1e-9
    check_rot_ew = norm(flip(permute(O, ((1,2),(4,6,5,7,3,8))), [4 7]) - O) < 1e-9
    check_flip_ud = norm(flip(permute(O, ((1,2),(6,4,5,3,7,8))), [3 6]) - O) < 1e-9
    check_flip_ns = norm(flip(permute(O, ((1,2),(3,7,5,6,4,8))), [4 7]) - O) < 1e-9
    check_flip_ew = norm(flip(permute(O, ((1,2),(3,4,8,6,7,5))), [5 8]) - O) < 1e-9

    check_imag = norm(imag.(O.data)) < 1e-9

    if i == 1
        @test !check_rot_ud && !check_rot_ns && !check_rot_ew && !check_flip_ud && !check_flip_ns && !check_flip_ew && check_imag
    else
        @test check_rot_ud && check_rot_ns && check_rot_ew && check_flip_ud && check_flip_ns && check_flip_ew && !check_imag
    end
end
