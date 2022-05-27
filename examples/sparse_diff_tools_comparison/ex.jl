import NonlinearEquations
import PyPlot
using SparseDiffTools

fcalls = 0
function f(y,x) # in-place
  global fcalls += 1
  for i in 2:length(x)-1
    y[i] = x[i-1] - 2x[i] + x[i+1]
  end
  y[1] = -2x[1] + x[2]
  y[end] = x[end-1] - 2x[end]
  nothing
end

@NonlinearEquations.equations function gw1d(h)
	NonlinearEquations.setnumequations(length(h))
	NonlinearEquations.addterm(1, -2 * h[1] + h[2])
	for i = 2:length(h) - 1
		NonlinearEquations.addterm(i, h[i - 1] - 2 * h[i] + h[i + 1])
	end
	NonlinearEquations.addterm(length(h), -2 * h[end] + h[end - 1])
end


using SparsityDetection, SparseArrays

using FiniteDiff
@show fcalls # 5

function jacobian!(input, jac)
	output = similar(input)
	sparsity_pattern = jacobian_sparsity(f,output,input)
	jac = Float64.(sparse(sparsity_pattern))
	colors = matrix_colors(jac)
	FiniteDiff.finite_difference_jacobian!(jac, f, input, colorvec=colors)
end

t_sdt = Float64[]
t_nle = Float64[]
#Ns = 2 .^ (10:20)
Ns = 2 .^ (10:24)
for N in Ns
	input = rand(N)
	output = similar(input)
	jacobian!(input, jac)
	t = @elapsed jacobian!(input, jac)
	t = min(t, @elapsed jacobian!(input, jac))
	t = min(t, @elapsed jacobian!(input, jac))
	@show t
	push!(t_sdt, t)
	gw1d_h(input)
	t = @elapsed gw1d_h(input)
	t = min(t, @elapsed gw1d_h(input))
	t = min(t, @elapsed gw1d_h(input))
	push!(t_nle, t)
	@show t
end

DelimitedFiles.writedlm("SparseDiffTools_github_example.dlm", hcat(Ns, t_sdt, t_nle))

fig, ax = PyPlot.subplots()
ax.loglog(Ns, t_sdt, ".", color="C0", label="SparseDiffTools.jl")
ax.loglog(Ns, t_nle, ".", color="C1", label="NonlinearEquations.jl")
ax.set(xlabel="Number of equations", ylabel="Time [s]")
ax.legend()
fig.savefig("laplacian.pdf")
display(fig)
println()
PyPlot.close(fig)
