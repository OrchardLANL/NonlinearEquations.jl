#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
using DiffEqSensitivity
import NonlinearEquations
using OrdinaryDiffEq
import PyPlot

include("utilities.jl")
include("inputdeck.jl")

@NonlinearEquations.equations exclude=(coords, dirichletnodes, neighbors, areasoverlengths) function req(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
	NonlinearEquations.setnumequations(length(psi))
	for i = 1:length(psi)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, psi[i] - dirichletpsis[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, hm(kr(psi[node1], alphas[i], Ns[i]), kr(psi[node2], alphas[i], Ns[i])) * Ks[i] * (psi[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, hm(kr(psi[node1], alphas[i], Ns[i]), kr(dirichletpsis[node2], alphas[i], Ns[i])) * Ks[i] * (dirichletpsis[node2] + coords[2, node2] - psi[node1] - coords[2, node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			end
		end
	end
end

function unpack(p)
	@assert length(p) == length(neighbors) + size(coords, 2)
	Ks = p[1:length(neighbors)]
	Qs = p[length(neighbors) + 1:end]
	return Ks, Qs
end

function f(u, p, t)
	Ks, Qs = unpack(p)
	return req_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
end

function jac(u, p, t)
	Ks, Qs = unpack(p)
	return req_psi(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
end

psi0 = fill(-5.0, size(coords, 2))
psi0[dirichletnodes] = dirichletpsis[dirichletnodes]
p = [Ks; Qs]
tspan = [0, 1e2]
odef = ODEFunction(f; jac=jac)
prob = ODEProblem(odef, psi0, tspan, p)
@time soln = solve(prob, ImplicitEuler())
plottransient(soln)
