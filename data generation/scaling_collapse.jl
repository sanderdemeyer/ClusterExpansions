using KrylovKit
using JLD2
using Zygote
using Plots

function rescale_correlation_length(Ts, ξs, δs, Tc, ν)
    xdata = [(T - Tc) * δ^(-1 / ν) for (T, δ) in zip(Ts, δs)]
    ydata = [log(ξ * δ) for (ξ, δ) in zip(ξs, δs)]
    return xdata, ydata
end

function error_measure(x1_sorted, y1_sorted, x, y)
    # Find the interval in x1 where x lies
    if x < x1_sorted[1] || x > x1_sorted[end]
        # x is outside the interpolation range
        return 0.0
    else
        # Find the index such that x1_sorted[j] <= x < x1_sorted[j+1]
        j = findfirst(j -> x1_sorted[j] <= x < x1_sorted[j+1], 1:length(x1_sorted)-1)

        # Linear interpolation
        x_low, x_high = x1_sorted[j], x1_sorted[j+1]
        y_low, y_high = y1_sorted[j], y1_sorted[j+1]

        t = (x - x_low) / (x_high - x_low)
        interpolated_y = (1 - t) * y_low + t * y_high
        return (interpolated_y - y)^2
    end
end

function error_measure_collaps(x1, y1, x2, y2)
    # Ensure inputs are sorted by x-values
    sorted_indices1 = sortperm(x1)
    x1_sorted = x1[sorted_indices1]
    y1_sorted = y1[sorted_indices1]

    sorted_indices2 = sortperm(x2)
    x2_sorted = x2[sorted_indices2]
    y2_sorted = y2[sorted_indices2]

    differences = [error_measure(x1_sorted, y1_sorted, x, y) for (x,y) in zip(x2_sorted, y2_sorted)]
    return sum(differences)
end

function error_measure_dataset(xdata, ydata, Ts_i, ξs_i, δs_i)
    xdata_i, ydata_i = rescale_correlation_length(Ts_i, ξs_i, δs_i, Tc, ν)
    return error_measure_collaps(xdata, ydata, xdata_i, ydata_i)

end

function collapse_correlation_function(cuts, Dcuts, name; Tc0 = 1.27, ν0 = 1.0, plotting = true)
    Ts = []
    ξs = []
    δs = []
    for (cut, Dcut) in zip(cuts, Dcuts)
        file = jldopen(name(cut, Dcut), "r")
        push!(Ts, file["Ts"])
        push!(ξs,file["ξs"])
        push!(δs,file["δs"])
        close(file)
    end
    println("Starting optimizing")
    (Tc_final, ν_final), cost_final = optimize((Tc0, ν0), LBFGS(4; verbosity = 3, maxiter = 100)) do (Tc, ν)
        E, gs = withgradient((Tc, ν)) do (Tc1, ν1)
            xdata, ydata = rescale_correlation_length(Ts[1], ξs[1], δs[1], Tc1, ν1)
            return sum([error_measure_dataset(xdata, ydata, Ts_i, ξs_i, δs_i) for (Ts_i, ξs_i, δs_i) in zip(Ts[2:end], ξs[2:end], δs[2:end])])
        end
        g = only(gs)
        return E, g
    end
    println("Result is (Tc = $(Tc_final), ν = $(ν_final)), cost = $cost_final")
    if plotting

        plt1 = scatter()
        plt2 = scatter()
        for (i,(Ts_i, ξs_i, δs_i)) in enumerate(zip(Ts, ξs, δs))
            xdata_i, ydata_i = rescale_correlation_length(Ts_i, ξs_i, δs_i, Tc_final, ν_final)
            scatter!(plt1, xdata_i, ydata_i, label = "cut = $(cuts[i]), Dcut = $(Dcuts[i])")
            scatter!(plt2, Ts, ξs_i, label = "cut = $(cuts[i]), Dcut = $(Dcuts[i])")
        end
        xlims!(plt1, (-250, 250))
        xlabel!(plt1, "(T - Tc) * δ^(-1/ν)")
        ylabel!(plt1, "log(ξ * δ)")
        title!(plt1, "Data collapse for Ising model with g = $(g)")
        display(plt1)
        xlabel!(plt2, "T")
        ylabel!(plt2, "ξ")
        title!(plt2, "Correlation length for Ising model with g = $(g)")
        display(plt2)


        plt1 = scatter()
        for (i,(Ts_i, ξs_i, δs_i)) in enumerate(zip(Ts, ξs, δs))
            xdata_i, ydata_i = rescale_correlation_length(Ts_i, ξs_i, δs_i, Tc0, ν0)
            scatter!(plt1, xdata_i, ydata_i, label = "cut = $(cuts[i]), Dcut = $(Dcuts[i])")
        end
        xlims!(plt1, (-25, 25))
        xlabel!(plt1, "(T - Tc) * δ^(-1/ν)")
        ylabel!(plt1, "log(ξ * δ)")
        title!(plt1, "Data collapse for Ising model with g = $(g) - initial")
        display(plt1)
    end
    return Tc_final, ν_final, cost_final
end

x1 = [5, 2, 1, 3]
y1 = [5, 3, 1, 6]
x2 = [2.5, 6, 1.5, 4]
y2 = [3.5, 5, 2, 6]
g = 2.5
name = (cut, Dcut) -> "Ising_critical_exponents_g_$(g)_cut_$(cut)_Dcut_$(Dcut).jld2"
differences = collapse_correlation_function([2.5 3.5], [4 6], name)