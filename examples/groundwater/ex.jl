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

#h0 = -ones(size(coords, 2))
#h0[dirichletnodes] = zeros(length(dirichletnodes))

function solveforh(Ks; doplot=false, donewtonish=true)
	res(h) = groundwater_residuals(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
	jac(h) = groundwater_h(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
	h0 = zeros(size(coords, 2))
	h = jac(h0) \ -res(h0)
	if doplot
		callback(h, res(h), jac(h), 0)
	end
	return h
end

h = solveforh(Ks; doplot=true)

#compute the gradient
Random.seed!(1)
obsnode = rand(1:length(h))
g(h, Ks) = h[obsnode]
function g_h(h, p)
	retval = zeros(length(h))
	retval[obsnode] = 1.0
	return retval
end
g_Ks(h, Ks) = zeros(length(Ks))
f_h(h, Ks) = groundwater_h(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
f_Ks(h, Ks) = groundwater_Ks(h, Ks, neighbors, areasoverlengths, dirichletnodes, dirichleths, Qs)
grad = NonlinearEquations.gradient(h, Ks, g_h, g_Ks, f_h, f_Ks)
plotgradient(grad, obsnode)

fdgrad = similar(grad)
sortedgradindices = sort(1:length(grad); by=i->abs(grad[i]), rev=true)
@time for i = 1:length(fdgrad)
	global fdgrad
	global Ks
	global h
	dk = 1e-8
	h0 = h
	Ks0 = copy(Ks)
	thisKs = copy(Ks)
	thisKs[i] += dk
	thish = solveforh(thisKs)
	fdgrad[i] = (thish[obsnode] - h0[obsnode]) / dk
end
plotgradient(fdgrad, obsnode)
@test isapprox(grad, fdgrad; rtol=1e-4)
