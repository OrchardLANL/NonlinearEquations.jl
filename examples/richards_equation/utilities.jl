using Test
import Calculus
import NonlinearEquations
import Random
import SparseArrays

function updateentries!(dest::SparseArrays.SparseMatrixCSC, src::SparseArrays.SparseMatrixCSC)
	if dest.colptr != src.colptr || dest.rowval != src.rowval
		error("Cannot update entries unless the two matrices have the same pattern of nonzeros")
	end
	copy!(dest.nzval, src.nzval)
end

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
	#=
	q = :(2.0 / (1 / $x + 1 / $y))
	@show q
	return :($esc(q))
	=#
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
		@show i
		fig, axs = PyPlot.subplots(2, 3, figsize=(20, 9))
		psi = reshape(psi, ns[2], ns[1])
		saturation = reshape(saturation, ns[2], ns[1])
		residuals = reshape(residuals, ns[2], ns[1])
		@show size(psi), extrema(psi)
		img = axs[1].imshow(map(x->x > 0 ? x : 0, psi), origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[1])
		img = axs[2].imshow(saturation, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[2])
		goodindices = filter(i->coords[2, i] == maxs[2], 1:size(coords, 2))
		img = axs[3].imshow(map(x->x < 0 ? x : 0, psi), origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[3])
		img = axs[4].imshow(residuals, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], maxs[2]])
		fig.colorbar(img, ax=axs[4])
		img = axs[5].plot(range(0, 1; length=length(J.nzval)), sort(abs.(J.nzval)), ".", alpha=0.1, ms=10)
		display(fig)
		println()
		println()
		PyPlot.close(fig)
	end
end

function plotgraph(neighbors, coords, Ks)
	fig, ax = PyPlot.subplots()
	ax.plot(coords[1, :], coords[2, :], "k.", ms=20)
	cmap = PyPlot.ColorMap("cool")
	minK, maxK = extrema(Ks)
	for (k, (i, j)) in enumerate(neighbors)
		ax.plot(coords[1, [i, j]], coords[2, [i, j]], alpha=0.5, color=cmap((Ks[k] - minK) / (maxK - minK)))
	end
	display(fig)
	println()
	println()
	PyPlot.close(fig)
end

function krmat(psi, sym)
	return kr(psi, params[sym][2:3]...)
end
krclay(psi) = krmat(psi, :clay)
krclaysilt(psi) = krmat(psi, :claysilt)
