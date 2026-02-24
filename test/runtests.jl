using SafeTestsets

@time @safetestset "Ground state search" begin
    include("ground_state_search.jl")
end

@time @safetestset "Phase transition - Ising model" begin
    include("phase_transition_ising.jl")
end

@time @safetestset "Phase transition - Spinless Fermion model" begin
    include("phase_transition_spinless_fermions.jl")
end

@time @safetestset "Utility functions" begin
    include("utility_functions.jl")
end
