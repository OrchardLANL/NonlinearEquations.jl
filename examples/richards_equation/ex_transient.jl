#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
using BenchmarkTools
using DiffEqSensitivity
import DifferentiableBackwardEuler
import NonlinearEquations
using OrdinaryDiffEq
#import PyPlot
import Zygote

#include("ex.jl")
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
	@assert length(p) == length(neighbors)
	Ks = p[1:length(neighbors)]
	return Ks
	#=
	@assert length(p) == length(neighbors) + size(coords, 2)
	Ks = p[1:length(neighbors)]
	Qs = p[length(neighbors) + 1:end]
	return Ks, Qs
	=#
end

function f(u, p, t)
	Ks = unpack(p)
	return req_residuals(u, Ks, Qs)
end

function f_u(u, p, t)
	Ks = unpack(p)
	return req_psi(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
end

function f_p(u, p, t)
	Ks = unpack(p)
	return req_Ks(u, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, specificstorage, volumes)
end

f_t(u, p, t) = zeros(length(u))

psi0 = fill(-5.0, size(coords, 2))
psi0[dirichletnodes] = dirichletpsis[dirichletnodes]
p = Ks
tspan = [0, 1e2]
odef = ODEFunction(f; jac=f_u, jac_prototype=f_u(psi0, p, 0.0))
prob = ODEProblem(odef, psi0, tspan, p)
@time soln_diffeq = solve(prob, ImplicitEuler())
@time soln_dbe = DifferentiableBackwardEuler.steps(psi0, f, f_u, f_p, f_t, p, tspan[1], tspan[2]; ftol=1e-12, method=:newton)
@show sum((soln_diffeq[:, :] .- soln_dbe) .^ 2)
@show sum(soln_diffeq[:, :] .^ 2)
obsnode = div(3 * length(psi0), 4)
g(p) = DifferentiableBackwardEuler.steps(psi0, f, f_u, f_p, f_t, p, tspan[1], tspan[2]; ftol=1e-12, method=:newton)[obsnode, end]
g(p)
@time g(p)
grad_zygote = Zygote.gradient(g, p)[1]
@time grad_zygote = Zygote.gradient(g, p)[1]

function naivegradient(f, x0s...; delta::Float64=1e-8)
	f0 = f(x0s...)
	gradient = map(x->zeros(size(x)), x0s)
	for j = 1:length(x0s)
		x0 = copy(x0s[j])
		for i = 1:length(x0)
			x = copy(x0)
			x[i] += delta
			xs = (x0s[1:j - 1]..., x, x0s[j + 1:length(x0s)]...)
			fval = f(xs...)
			gradient[j][i] = (fval - f0) / delta
		end
	end
	return gradient
end

function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8)
	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
	indicestocheck = [indicestocheck; rand(1:length(x0), n)]
	f0 = f(x0)
	for i in indicestocheck
		x = copy(x0)
		x[i] += delta
		fval = f(x)
		grad_f_i = (fval - f0) / delta
		@show i
		@show grad_f_i
		@show gradf[i]
	end
end
checkgradientquickly(g, p, grad_zygote, 5; delta=1e-4)

#@show length(p)
#@time grad_naive = naivegradient(g, p)[1]
#@show sqrt(sum((grad_zygote .- grad_naive) .^ 2) / sum(grad_naive .^ 2))
#@show sum(grad_naive .^ 2)
