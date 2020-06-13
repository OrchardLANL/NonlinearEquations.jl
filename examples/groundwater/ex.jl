import NLsolve
import NonlinearEquations
import PyPlot
import Random

include("utilities.jl")
include("inputdeck.jl")

@NonlinearEquations.equations exclude=(dirichletnodes, neighbors, areasoverlengths) function groundwater(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
	NonlinearEquations.setnumequations(length(h))
	for i = 1:length(h)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, h[i] - dirichleths[i])
		else
			NonlinearEquations.addterm(i, Qs[i])
		end
	end
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, Ks[i] * (h[node2] - h[node1]) * areasoverlengths[i])
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, Ks[i] * (dirichleths[node2] - h[node1]) * areasoverlengths[i])
			end
		end
	end
end

psi0 = -ones(size(coords, 2))
psi0[dirichletnodes] = zeros(length(dirichletnodes))

function solveforpsi(Ks, psi0=psi0; doplot=false, donewtonish=true)
	res(psi) = groundwater_residuals(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, Qs)
	jac(psi) = groundwater_h(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, Qs)
	j!(J, psi) = groundwater_h!(J, psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, Qs)
	f!(residuals, psi) = copy!(residuals, res(psi))
	psi1 = psi0
	#callback(psi1, res(psi1), jac(psi1), 0)
	df = NLsolve.OnceDifferentiable(f!, j!, psi1, res(psi0), jac(psi0))
	nls = NLsolve.nlsolve(df, psi1; show_trace=false, iterations=200, ftol=1e-15)
	if doplot
		callback(nls.zero, res(nls.zero), jac(nls.zero), 0)
	end
	return nls.zero
end

psi = solveforpsi(Ks; doplot=true)

#compute the gradient
Random.seed!(1)
obsnode = rand(1:length(psi))
g(psi, Ks) = psi[obsnode]
function g_h(psi, p)
	retval = zeros(length(psi))
	retval[obsnode] = 1.0
	return retval
end
g_Ks(psi, Ks) = zeros(length(Ks))
f_h(psi, Ks) = groundwater_h(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, Qs)
f_Ks(psi, Ks) = groundwater_Ks(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, Qs)
grad = NonlinearEquations.gradient(psi, Ks, g_h, g_Ks, f_h, f_Ks)
plotgradient(grad, obsnode)

fdgrad = similar(grad)
sortedgradindices = sort(1:length(grad); by=i->abs(grad[i]), rev=true)
@time for i = 1:length(fdgrad)
	global fdgrad
	global Ks
	global psi
	dk = 1e-8
	psi0 = psi
	Ks0 = copy(Ks)
	thisKs = copy(Ks)
	thisKs[i] += dk
	thispsi = solveforpsi(thisKs, psi0; donewtonish=false)
	fdgrad[i] = (thispsi[obsnode] - psi0[obsnode]) / dk
end
plotgradient(fdgrad, obsnode)
@test isapprox(grad, fdgrad; rtol=1e-4)
