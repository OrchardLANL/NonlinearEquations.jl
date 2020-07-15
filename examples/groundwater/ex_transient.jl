using DiffEqSensitivity
import NLsolve
import NonlinearEquations
using OrdinaryDiffEq
import Random
import Zygote

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

groundwater_residuals(us, Ks) = groundwater_residuals(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
groundwater_h(us, Ks) = groundwater_h(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
groundwater_Ks(us, Ks) = groundwater_Ks(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)

#=
@Zygote.adjoint function groundwater_residuals(us, Ks)
	residuals = groundwater_residuals(us, Ks)
	back = delta->begin
		retval = ((delta' * groundwater_h(us, Ks))', (delta' * groundwater_Ks(us, Ks))')
		return retval
	end
	return residuals, back
end
=#

function unpack(p)
	@assert length(p) == length(neighbors)
	Ks = p[1:length(neighbors)]
	return Ks
	#@assert length(p) == length(neighbors) + size(coords, 2)
	#Ks = p[1:length(neighbors)]
	#Qs = p[length(neighbors) + 1:end]
	#return Ks, Qs
end

function f(u, p, t)
	Ks = unpack(p)
	return groundwater_residuals(u, Ks)
end

@Zygote.adjoint f(u, p, t) = f(u, p, t), delta->(jac(u, p, t)' * delta, paramjac(u, p, t)' * delta, zeros(length(u)))

function jac(u, p, t)
	Ks = unpack(p)
	return groundwater_h(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end

function paramjac(u, p, t)
	Ks = unpack(p)
	return groundwater_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs, specificstorage, volumes)
end

h0 = zeros(size(coords, 2))
h0[dirichletnodes] = dirichleths[dirichletnodes]
p = Ks
tspan = [0, 1e1]
odef = ODEFunction(f; jac=jac, jac_prototype=jac(h0, p, 0.0), paramjac=paramjac)
prob = ODEProblem(odef, h0, tspan, p)
soln = solve(prob, ImplicitEuler())
#plottransient(soln)
function g(p)
	prob = ODEProblem(odef, h0, tspan, p)
	#sensealg = InterpolatingAdjoint()
	sensealg = QuadratureAdjoint()
	#sensealg = TrackerAdjoint()
	return sum(Array(solve(prob, ImplicitEuler(); u0=prob.u0, p=prob.p, sensealg=sensealg))[:, end])
end
#=
print("function eval: ")
@btime $g($p)
print("zygote gradient: ")
zg = @btime Zygote.gradient($g, $p)[1]
=#
g(p)
t1 = @elapsed g(p)
@show t1
Zygote.gradient(g, p)
t2 = @elapsed Zygote.gradient(g, p)[1]
@show t2
speedup = t1 * length(p) / t2
@show speedup
@show length(p)
