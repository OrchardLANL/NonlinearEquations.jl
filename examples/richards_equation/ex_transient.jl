#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
using BenchmarkTools
using DiffEqSensitivity
import NonlinearEquations
using OrdinaryDiffEq
#import PyPlot
import Zygote

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

req_residuals(us, Ks, Qs) = req_residuals(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
req_psi(us, Ks, Qs) = req_psi(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
req_Ks(us, Ks, Qs) = req_Ks(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
req_Qs(us, Ks, Qs) = req_Qs(us, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)

@Zygote.adjoint function req_residuals(us, Ks, Qs)
	residuals = req_residuals(us, Ks, Qs)
	back = delta->begin
		retval = ((delta' * req_psi(us, Ks, Qs))', (delta' * req_Ks(us, Ks, Qs))', (delta' * req_Qs(us, Ks, Qs))')
		return retval
	end
	return residuals, back
end

function unpack(p)
	@assert length(p) == length(neighbors) + size(coords, 2)
	Ks = p[1:length(neighbors)]
	Qs = p[length(neighbors) + 1:end]
	return Ks, Qs
end

function f(u, p, t)
	Ks, Qs = unpack(p)
	return req_residuals(u, Ks, Qs)
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
animatetransient(soln)
#=
function g(p)
	prob = ODEProblem(odef, psi0, tspan, p)
	return sum(Array(solve(prob, ImplicitEuler(); u0=prob.u0, p=prob.p))[:, end])
end
#=
print("function eval: ")
@btime $g($p)
print("zygote gradient: ")
zg = @btime Zygote.gradient($g, $p)[1]
=#
@time g(p)
@time g(p)
zg = @time Zygote.gradient(g, p)[1]
nothing
=#
