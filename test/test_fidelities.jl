using TensorKit
using MPSKitModels
using ClusterExpansions
using PEPSKit
using Plots
using JLD2

function _stack_pepos(pepos)
    return InfinitePEPO(cat(pepos...; dims = 3))
end

fermionic = true
if fermionic
    param = (t, V, μ) = (1.0, -2.0, 0.0)
    model = spinless_fermion_operators
else
    param = (J, g, z) = (1.0, 0.0, 0.0)
    model = ising_operators
end

ce_kwargs = ()
ce_alg = model(param...; ce_kwargs...)
β = 0.1
O = evolution_operator(ce_alg, β; canoc_alg = nothing)
A₁ = A₂ = O
A = (A₁, A₂)
maxiterations = 30

Dcut = 3
χenv_approx = 5
# envspace = envspace_approx = Vect[fℤ₂](0 => div(χenv_approx, 2), 1 => div(χenv_approx, 2))
# trunc_space = Vect[fℤ₂](0 => div(Dcut, 2), 1 => div(Dcut, 2))
envspace = envspace_approx = Vect[fℤ₂](0 => 3, 1 => 2)
trunc_space = Vect[fℤ₂](0 => 3, 1 => 2)

ctm_alg = SimultaneousCTMRG(verbosity = 2; maxiter = 750)
ctm_alg_verbose = SimultaneousCTMRG(verbosity = 3; maxiter = 500, tol = 1e-7)

# Truncate locally
trunc_alg = NoEnvTruncation(truncdim(Dcut))
B, _ = approximate_state(A, trunc_alg)

# Initialize environments
network_A = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(A₂), PEPSKit._dag(A₁))))
network_B = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
network_overlap = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))

# Update environments
env_A, = leading_boundary(CTMRGEnv(network_A, envspace), network_A, ctm_alg_verbose)
env_B, = leading_boundary(CTMRGEnv(network_B, envspace), network_B, ctm_alg)
env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)

# Calculate local fidelity
ϵ_local = abs(network_value(network_overlap, env_overlap) / sqrt(network_value(network_A, env_A) * network_value(network_B, env_B)))

# Truncate globally
trunc_alg = ApproximateEnvTruncation(ctm_alg, envspace_approx, truncdim(Dcut); tol = 0.0, maxiter = maxiterations)
Bs, _ = ClusterExpansions.approximate_state_testing(A, trunc_alg)

ϵs_global = []
for B = Bs
    # Initialize environments
    network_B = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    network_overlap = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))

    # Update environments
    env_B, = leading_boundary(CTMRGEnv(network_B, envspace), network_B, ctm_alg)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)

    # Calculate global fidelity
    push!(ϵs_global, abs(network_value(network_overlap, env_overlap) / sqrt(network_value(network_A, env_A) * network_value(network_B, env_B))))
end

# Truncate variationally
trunc_alg = VOPEPO(ctm_alg, envspace, trunc_space, maxiterations, maxiterations; verbosity = 2, check_fidelity = false)
env_double, env_triple = ClusterExpansions.initialize_vomps_environments(domain(O)[1], domain(O)[1], trunc_alg)
B, env_double, env_triple, ϵs_VOPEPO, Bs = approximate_state(A, env_double, env_triple, trunc_alg)

ϵs_var = []
for B = Bs
    # Initialize environments
    network_B = InfiniteSquareNetwork(_stack_pepos((B, PEPSKit._dag(B))))
    network_overlap = InfiniteSquareNetwork(_stack_pepos((A₁, A₂, PEPSKit._dag(B))))

    # Update environments
    env_B, = leading_boundary(CTMRGEnv(network_B, envspace), network_B, ctm_alg)
    env_overlap, = leading_boundary(CTMRGEnv(network_overlap, envspace), network_overlap, ctm_alg)

    # Calculate variational fidelity
    push!(ϵs_var, abs(network_value(network_overlap, env_overlap) / sqrt(network_value(network_A, env_A) * network_value(network_B, env_B))))
end
file = jldopen("check_fidelities_V_$(V)_Dcut_$(Dcut)_χenv_$(χenv_approx).jld2", "w")
file["ϵ_local"] = ϵ_local
file["ϵs_global"] = ϵs_global
file["ϵs_var"] = ϵs_var
close(file)

plt = plot(abs.(1 .- ϵs_global), label = "global optimization", yaxis=:log)
plot!(abs.(1 .- ϵs_var), label = "variational optimization", yaxis=:log)
hline!([abs(1-ϵ_local)], label = "local optimization")
xlabel!("Iteration")
ylabel!("1-fidelity")
display(plt)
