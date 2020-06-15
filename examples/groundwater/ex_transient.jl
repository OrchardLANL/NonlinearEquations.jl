using DiffEqSensitivity
import NLsolve
import NonlinearEquations
using OrdinaryDiffEq
import Random

include("utilities.jl")
include("inputdeck.jl")

@NonlinearEquations.equations exclude=(dirichletnodes, neighbors, areasoverlengths) function groundwater(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
	NonlinearEquations.setnumequations(length(h))
	for i = 1:length(h)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, h[i] - dirichleths[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (specificstorage[i] * volumes[i]))
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, Ks[i] * (h[node2] - h[node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, Ks[i] * (dirichleths[node2] - h[node1]) * areasoverlengths[i] / (specificstorage[node1] * volumes[node1]))
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
	return groundwater_residuals(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end

function jac(u, p, t)
	Ks, Qs = unpack(p)
	return groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end

h0 = zeros(size(coords, 2))
h0[dirichletnodes] = dirichleths[dirichletnodes]
p = [Ks; Qs]
tspan = [0, 1e2]
odef = ODEFunction(f; jac=jac)
prob = ODEProblem(odef, h0, tspan, p)
soln = solve(prob, ImplicitEuler())
plottransient(soln)
