using Test
import Calculus
import NonlinearEquations
import Random
import SparseArrays

function kr(psi, alpha, N)
	if psi < 0
		m = (N - 1) / N
		denom = 1 + abs(alpha * psi) ^ N
		numer = 1 - abs(alpha * psi) ^ (N - 1) * denom ^ (-m)
		return numer ^ 2 / denom ^ (m / 2)
	else
		return 1.0
	end
end

function hm(x, y)
	return 2 / (1 / x + 1 / y)
end

function Calculus.differentiate(x::Calculus.SymbolParameter{:abs}, args, wrt)
	if length(args) > 1
		error("Too many arguments passed to abs()")
	end
	arg = args[1]
	return :(ifelse($arg > 0, 1, -1) * $(Calculus.differentiate(arg, wrt)))
end

macro kr(psi, alpha, N)
	q = :(ifelse($(esc(psi)) < 0, (1 - abs($(esc(alpha)) * $(esc(psi))) ^ ($(esc(N)) - 1) * (1 + abs($(esc(alpha)) * $(esc(psi))) ^ $(esc(N))) ^ -(($(esc(N)) - 1) / $(esc(N)))) ^ 2 / (1 + abs($(esc(alpha)) * $(esc(psi))) ^ $(esc(N))) ^ (($(esc(N)) - 1) / (2 * $(esc(N)))), 1.0))
	return NonlinearEquations.escapesymbols(q, [:ifelse, :abs, :+, :-, :^, :/, :<, :*])
end

macro kr2(sat, psi, alpha, N)
	q = :(ifelse($(esc(psi)) < 0, (1 - abs($(esc(alpha)) * $(esc(psi))) ^ ($(esc(N)) - 1) * abs($(esc(sat)))) ^ 2 * sqrt(abs($(esc(sat)))), 1.0))
	return NonlinearEquations.escapesymbols(q, [:ifelse, :abs, :+, :-, :^, :/, :<, :*, :sqrt])
end

macro hm(x, y)
	q = :(2.0 / (1 / $(esc(x)) + 1 / $(esc(y))))
	return NonlinearEquations.escapesymbols(q, [:/ :+])
end

Random.seed!(0)
for i = 1:10 ^ 3
	psi = randn()
	alpha = randn()
	N = randn()
	@test kr(psi, alpha, N) == @kr(psi, alpha, N)
	x = randn()
	y = randn()
	@test hm(x, y) == @hm(x, y)
	K = randn()
	psi2 = rand()
	alpha2 = randn()
	N2 = randn()
	K2 = rand()
	@test hm(kr(psi, alpha, N) * K, kr(psi2, alpha2, N2) * K2) == @hm(@kr(psi, alpha, N) * K, @kr(psi2, alpha2, N2) * K2)
end

function effective_saturation(alpha::Number, psi::Number, N::Number)
	m = (N - 1) / N
	if psi < 0
		return (1 + abs(alpha * psi) ^ N) ^ (-m)
	else
		return 1
	end
end

function saturation(psi::Array, coords::Array)
	Se = similar(psi)
	for i = 1:length(psi)
		if inclay(coords[1, i], coords[2, i])
			_, alpha, N, residual_saturation = params[:clay]
		else
			_, alpha, N, residual_saturation = params[:claysilt]
		end
		Se[i] = residual_saturation + effective_saturation(alpha, psi[i], N) * (1 - residual_saturation)
	end
	return Se
end

function callback(psi, residuals, J, i, saturation=saturation(psi, coords))
	if mod(i, 10) == 0
		fig, axs = PyPlot.subplots(2, 2, figsize=(16, 9))
		psi = reshape(psi, ns[2], ns[1])
		saturation = reshape(saturation, ns[2], ns[1])
		residuals = reshape(residuals, ns[2], ns[1])
		img = axs[1].imshow(map(x->x > 0 ? x : 0, psi), origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[1])
		img = axs[2].imshow(saturation, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[2])
		goodindices = filter(i->coords[2, i] == maxs[2], 1:size(coords, 2))
		img = axs[3].imshow(map(x->x < 0 ? x : 0, psi), origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[3])
		img = axs[4].imshow(residuals, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[4])
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
