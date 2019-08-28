#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
import NLsolve
import NonlinearEquations
import PyPlot
import Random

include("utilities.jl")
include("inputdeck.jl")

@NonlinearEquations.equations exclude=(coords, dirichletnodes, neighbors, areasoverlengths) function reqnumenv(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
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

@NonlinearEquations.equations exclude=(coords, dirichletnodes, neighbors, areasoverlengths) function req(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
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

psi0 = -ones(size(coords, 2))
psi0[dirichletnodes] = zeros(length(dirichletnodes))

function solveforpsi(Ks, psi0=psi0; doplot=false, donewtonish=true)
	resnumenv(psi) = reqnumenv_residuals(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	jacnumenv(psi) = reqnumenv_psi(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	res(psi) = req_residuals(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	jac(psi) = req_psi(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	j!(J, psi) = req_psi!(J, psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	f!(residuals, psi) = copy!(residuals, res(psi))
	if donewtonish
		psi1 = NonlinearEquations.newtonish(resnumenv, jacnumenv, psi0; numiters=10, rate=0.05)
	else
		psi1 = psi0
	end
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
function g_psi(psi, p)
	retval = zeros(length(psi))
	retval[obsnode] = 1.0
	return retval
end
g_Ks(psi, Ks) = zeros(length(Ks))
f_psi(psi, Ks) = req_psi(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
f_Ks(psi, Ks) = req_Ks(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
grad = NonlinearEquations.gradient(psi, Ks, g_psi, g_Ks, f_psi, f_Ks)
grad = NonlinearEquations.gradient(psi, Ks, g_psi, g_Ks, f_psi, f_Ks)
plotgradient(grad, obsnode)

#=
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
=#
