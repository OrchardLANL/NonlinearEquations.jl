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
				asdf = @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(psi[node2], alphas[i], Ns[i]))
				NonlinearEquations.addterm(node1, asdf * Ks[i] * (psi[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
				#NonlinearEquations.addterm(node1, @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(psi[node2], alphas[i], Ns[i])) * Ks[i] * (psi[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				asdf = @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(dirichletpsis[node2], alphas[i], Ns[i]))
				NonlinearEquations.addterm(node1, asdf * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
				#NonlinearEquations.addterm(node1, @hm(@kr(psi[node1], alphas[i], Ns[i]), @kr(dirichletpsis[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i])
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
j!(J, psi) = updateentries!(J, jac(psi))

@time begin
psi0 = -ones(size(coords, 2))
psi0[dirichletnodes] = zeros(length(dirichletnodes))
@time psi1 = NonlinearEquations.newtonish(resnumenv, jacnumenv, psi0; numiters=100, callback=callback, rate=0.05)

df = NLsolve.OnceDifferentiable(f!, j!, psi1, res(psi0), jac(psi0))
#nls = NLsolve.nlsolve(df, psi1; show_trace=true, method=:newton, linesearch=LineSearches.HagerZhang(), iterations=100)
nls = NLsolve.nlsolve(df, psi1; show_trace=true, iterations=100)
callback(nls.zero, res(nls.zero), jac(nls.zero), 0)
end

fig, ax = PyPlot.subplots()
ax.plot(sort(psi1), krclay.(sort(psi1)), label="clay")
ax.plot(sort(psi1), krclaysilt.(sort(psi1)), label="claysilt")
ax.legend()
display(fig)
println()
println()
PyPlot.close(fig)

#=
@NonlinearEquations.equations function req_impsat(psi_n_sat, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	NonlinearEquations.setnumequations(length(psi_n_sat))
	for i = 1:div(length(psi_n_sat), 2)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, psi_n_sat[i] - dirichletpsis[i])
		else
			NonlinearEquations.addterm(i, Qs[i])
		end
		NonlinearEquations.addterm(i + div(length(psi_n_sat), 2), psi_n_sat[i + div(length(psi_n_sat), 2)] - (1 + abs(alphas[i] * psi_n_sat[i]) ^ Ns[i]) ^ -((Ns[i] - 1) / Ns[i]))
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			sn1 = div(length(psi_n_sat), 2) + node1
			sn2 = div(length(psi_n_sat), 2) + node2
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, @hm(@kr2(psi_n_sat[sn1], psi_n_sat[node1], alphas[i], Ns[i]), @kr2(psi_n_sat[sn2], psi_n_sat[node2], alphas[i], Ns[i])) * Ks[i] * (psi_n_sat[node2] + coords[2, node2] - psi_n_sat[node1] - coords[2, node1]) * areasoverlengths[i])
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, @hm(@kr2(psi_n_sat[sn1], psi_n_sat[node1], alphas[i], Ns[i]), @kr2(psi_n_sat[sn2], psi_n_sat[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi_n_sat[node1] - coords[2, node1]) * areasoverlengths[i])
			end
		end
	end
end

res_impsat(psi_n_sat) = req_impsat_residuals(psi_n_sat, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
jac_impsat(psi_n_sat) = req_impsat_jacobian(psi_n_sat, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
f_impsat!(residuals, psi_n_sat) = copy!(residuals, res_impsat(psi_n_sat))
j_impsat!(J, psi_n_sat) = updateentries!(J, jac_impsat(psi_n_sat))

psi0 = [-ones(size(coords, 2)); fill(0.2, size(coords, 2))]
psi0[dirichletnodes] = zeros(length(dirichletnodes))

df = NLsolve.OnceDifferentiable(f_impsat!, j_impsat!, psi0, res_impsat(psi0), jac_impsat(psi0))
#nls = NLsolve.nlsolve(df, psi1; show_trace=true, method=:newton, linesearch=LineSearches.HagerZhang(), iterations=100)
nls = NLsolve.nlsolve(df, psi0; show_trace=true, iterations=100)
callback(nls.zero[1:size(coords, 2)], res_impsat(nls.zero)[1:size(coords, 2)], jac_impsat(nls.zero), 0, nls.zero[size(coords, 2) + 1:end])
=#

nothing
