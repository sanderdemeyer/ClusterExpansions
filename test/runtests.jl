using SafeTestsets

@time @safetestset "Canonical form" begin
    include("canonical_form.jl")
end

@time @safetestset "Correlation functions" begin
    include("correlation_functions.jl")
end

@time @safetestset "Exact exponential" begin
    include("exact_exponential.jl")
end

@time @safetestset "Ground state search" begin
    include("ground_state_search.jl")
end

@time @safetestset "Phase transitions" begin
    include("phase_transitions.jl")
end

# @time @safetestset "Truncation schemes" begin
#     include("truncation_schemes.jl")
# end

@time @safetestset "Utility functions" begin
    include("utility_functions.jl")
end
