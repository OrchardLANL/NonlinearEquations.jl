#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
import LineSearches
import NLsolve
import NonlinearEquations
import PyPlot

include("utilities.jl")
include("inputdeck.jl")

@NonlinearEquations.equations function reqnumenv(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	NonlinearEquations.setnumequations(length(psi))
	for i = 1:length(psi)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, psi[i] - dirichletpsis[i])
		else
			NonlinearEquations.addterm(i, Qs[i])
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				relperm = @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(psi[node2], alphas[i], Ns[i]))
				NonlinearEquations.addterm(node1, relperm * Ks[i] * (psi[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				relperm = @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(dirichletpsis[node2], alphas[i], Ns[i]))
				NonlinearEquations.addterm(node1, relperm * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
			end
		end
	end
end

resnumenv(psi) = reqnumenv_residuals(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
jacnumenv(psi) = reqnumenv_jacobian(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)

@NonlinearEquations.equations function req(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	NonlinearEquations.setnumequations(length(psi))
	for i = 1:length(psi)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, psi[i] - dirichletpsis[i])
		else
			NonlinearEquations.addterm(i, Qs[i])
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(psi[node2], alphas[i], Ns[i])) * Ks[i] * (psi[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(dirichletpsis[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
			end
		end
	end
end

res(psi) = req_residuals(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
jac(psi) = req_jacobian(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
f!(residuals, psi) = copy!(residuals, res(psi))
j!(J, psi) = NonlinearEquations.updateentries!(J, jac(psi))

@time begin
	psi0 = -ones(size(coords, 2))
	psi0[dirichletnodes] = zeros(length(dirichletnodes))
	@time psi1 = NonlinearEquations.newtonish(resnumenv, jacnumenv, psi0; numiters=10, rate=0.05)
	#callback(psi1, res(psi1), jac(psi1), 0)
	df = NLsolve.OnceDifferentiable(f!, j!, psi1, res(psi0), jac(psi0))
	nls = NLsolve.nlsolve(df, psi1; show_trace=false, iterations=200, ftol=1e-15)
end
callback(nls.zero, res(nls.zero), jac(nls.zero), 0)

#=
fig, ax = PyPlot.subplots()
ax.plot(sort(psi1), krclay.(sort(psi1)), label="clay")
ax.plot(sort(psi1), krclaysilt.(sort(psi1)), label="claysilt")
ax.legend()
display(fig)
println()
println()
PyPlot.close(fig)
=#
nothing
