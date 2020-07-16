import DifferentiableBackwardEuler
import NLsolve
import NonlinearEquations
using OrdinaryDiffEq
import Random
import Zygote

include("utilities.jl")
include("inputdeck.jl")
include("ex.jl")

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

groundwater_residuals(us, Ks) = groundwater_residuals(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
groundwater_h(us, Ks) = groundwater_h(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
groundwater_Ks(us, Ks) = groundwater_Ks(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)

function unpack(p)
	@assert length(p) == length(neighbors)
	Ks = p[1:length(neighbors)]
	return Ks
end

function f(u, p, t)
	Ks = unpack(p)
	return groundwater_residuals(u, Ks)
end

function f_u(u, p, t)
	Ks = unpack(p)
	return groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end

function f_p(u, p, t)
	Ks = unpack(p)
	return groundwater_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end

f_t(u, p, t) = zeros(length(u))

h0 = zeros(size(coords, 2))
h0[dirichletnodes] = dirichleths[dirichletnodes]
p = Ks
tspan = [0, 1e4]
odef = ODEFunction(f; jac=f_u, jac_prototype=f_u(h0, p, 0.0))
prob = ODEProblem(odef, h0, tspan, p)
@time soln_diffeq = solve(prob, ImplicitEuler())
@time soln_dbe = DifferentiableBackwardEuler.steps(h0, f, f_u, f_p, f_t, p, soln_diffeq.t; ftol=1e-12)
sum((soln_diffeq[:, :] .- soln_dbe) .^ 2)
@test isapprox(soln_diffeq[:, :], soln_dbe)
g(p) = DifferentiableBackwardEuler.steps(h0, f, f_u, f_p, f_t, p, soln_diffeq.t; ftol=1e-12)[obsnode, end]
@time dgdp_zygote = Zygote.gradient(g, p)[1]
@test isapprox(dgdp_zygote, grad)#this checks the zygote gradient against the steady state gradient that doesn't use zygote
