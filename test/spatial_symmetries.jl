using TensorKit
using ClusterExpansions
using PEPSKit
using TensorKitTensors
using Test
using Random

Random.seed!(2718281828)

# Set up the Heisenberg model
(Jx, Jy, Jz, z) = (1.0, -1.0, 1.0, 0.5)

# Parameters in the truncation scheme
Dcut = 4
trunc_alg = NoEnvTruncation(truncrank(Dcut))
χenv = 16 # Environment bond dimension used in the calculation of expectation values
β = 0.05

p = 4
spin_symmetry = Trivial

symmetries = [nothing, "C4v"]
svds = [true false]
Ts = [BigFloat, Complex{BigFloat}]

@testset "Check rotational invariance and imaginary component for p = $p, symmetry = $(symmetry)" for (i,(symmetry,svd,T)) in enumerate(zip(symmetries,svds,Ts))
        ce_algs = [
                ising_operators(1.0, 2.0, 0.0; spin_symmetry, p, T, symmetry, svd)
                ising_operators(-1.0, 2.0, 0.0; spin_symmetry, p, T, symmetry, svd)
                heisenberg_operators(1.0, -1.0, 1.0, 0.5; spin_symmetry, p, T, symmetry, svd)
        ]
        for (j,ce_alg) in enumerate(ce_algs)
                O = evolution_operator(ce_alg, β)

                check_rot = norm(flip(permute(O, ((1,2),(4,5,6,3))), [4 6]) - O) < 1e-7
                check_flip = norm(permute(O, ((1,2),(4,3,6,5))) - O) < 1e-7
                check_imag = norm(imag.(O.data)) < 1e-7

                @test (i == 1 && check_imag) || (i == 2 && check_rot && check_flip)
        end
end
