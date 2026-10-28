using SafeTestsets

# disabled because it takes too long
# @time @safetestset "Ground state search" begin 
#     include("ground_state_search.jl")
# end

@time @safetestset "Phase transition - Ising model" begin
    include("phase_transition_ising.jl")
end

@time @safetestset "Phase transition - Spinless Fermion model" begin
    include("phase_transition_spinless_fermions.jl")
end

@time @safetestset "Utility functions" begin
    include("utility_functions.jl")
end

@time @safetestset "Simple update - cubic lattice" begin
    include("simple_update_cubic.jl")
end

@time @safetestset "Finite-temperature simple update - cubic lattice" begin
    include("thermal_simple_update_cubic.jl")
end

@time @safetestset "Classical Ising: exact D = 2 simple update - cubic lattice" begin
    include("classical_ising_simple_update_cubic.jl")
end

@time @safetestset "Symmetries" begin
    include("spatial_symmetries.jl")
    include("internal_symmetries.jl")
    include("symmetries.jl")
end

@time @safetestset "Order of cluster expansion" begin
    include("exact_exponential.jl")
end

@time @safetestset "3D CTMRG" begin
    include("3d_symmetries.jl")
end
