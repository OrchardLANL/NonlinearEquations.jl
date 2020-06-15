using Test
using AbstractPlotting
import Calculus
using GLMakie; GLMakie.activate!()
using MakieLayout
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

function plottransient(soln)
	hend = soln(tspan[end])
	scene, layout = layoutscene(resolution=(1200, 600))
	t_slider = layout[3, 1:3] = LSlider(scene; range=range(0, tspan[end]; length=500), startvalue=0)
	t_observable = lift(t->[t, t], t_slider.attributes[:value])
	h_observable = lift(t->reshape(soln(t), ns[2], ns[1])', t_slider.attributes[:value])
	residuals_observable = lift(t->reshape(f(soln(t), p, t), ns[2], ns[1])', t_slider.attributes[:value])
	ax1 = layout[1, 1] = LAxis(scene; title="Head")
	ax2 = layout[1, 2] = LAxis(scene; title="Residuals")
	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	heatmap!(ax1, xs, ys, h_observable; colorrange=extrema(hend))
	heatmap!(ax2, xs, ys, residuals_observable, colorrange=extrema(hend))
	for ax in [ax1, ax2]
		xlims!(ax, mins[1], maxs[1])
		ylims!(ax, mins[2], maxs[2])
	end
	cbar = layout[1, 3] = LColorbar(scene; limits=extrema(hend), width=30)
	timeplot = layout[2, :] = LAxis(scene; title="time", xgridvisible=false, ygridvisible=false, yticklabelsvisible=false, yticksvisible=false)
	timeplot.height=20
	lines!(timeplot, t_observable, [0, 1], linewidth=5)
	ylims!(timeplot, 0, 1)
	xlims!(timeplot, 0, tspan[end])
	display(scene)
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
