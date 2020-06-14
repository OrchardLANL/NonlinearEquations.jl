using Test
import Calculus
import NonlinearEquations
import Random
import SparseArrays

function callback(psi, residuals, J, i)
	if mod(i, 10) == 0
		fig, axs = PyPlot.subplots(1, 2, figsize=(16, 9))
		psi = reshape(psi, ns[2], ns[1])
		residuals = reshape(residuals, ns[2], ns[1])
		img = axs[1].imshow(map(x->x > 0 ? x : 0, psi), origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[1])
		goodindices = filter(i->coords[2, i] == maxs[2], 1:size(coords, 2))
		img = axs[2].imshow(residuals, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[2])
		display(fig)
		println()
		println()
		PyPlot.close(fig)
	end
end

function i2i1i2(i)
	i2 = 1 + mod(i - 1, ns[2])
	i1 = 1 + div(i - i2, ns[2])
	return i1, i2
end

function updateneighborarrays!(a1, a2, pair, value)
	n1, n2 = pair
	if i2i1i2(n1)[1] == i2i1i2(n2)[1]#the neighbors have the same x coordinate, update a2
		i1, i2_1 = i2i1i2(n1)
		i2_2 = i2i1i2(n2)[2]
		@assert i2_1 < i2_2
		a2[i2_1, i1] = value
	else#the neighbors have the same y coordinate, update a1
		i1_1, i2 = i2i1i2(n1)
		i1_2 = i2i1i2(n2)[1]
		@assert i1_1 < i1_2
		a1[i2, i1_1] = value
	end
end

function breakupgradient(grad)
	grad1 = zeros(ns[2], ns[1] - 1)
	grad2 = zeros(ns[2] - 1, ns[1])
	for (v, pair) in zip(grad, neighbors)
		updateneighborarrays!(grad1, grad2, pair, v)
	end
	return grad1, grad2
end

function plotgradient(grad, obsnode; vmin=minimum(grad), vmax=maximum(grad))
	grad1, grad2 = breakupgradient(grad)
	fig, axs = PyPlot.subplots(1, 2, figsize=(16, 9))
	img = axs[1].imshow(grad1, extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
	fig.colorbar(img, ax=axs[1])
	img = axs[2].imshow(grad2, extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
	fig.colorbar(img, ax=axs[2])
	display(fig)
	println()
	println()
	PyPlot.close(fig)
end
