import DelimitedFiles
import DifferentiableBackwardEuler
import NonlinearEquations
import PyPlot
import SparseArrays
import Zygote

@NonlinearEquations.equations exclude=(t, N, t2index, Bi) function icewedge(h, fvals, h_t, Bi, t, N, t2index)
	NonlinearEquations.setnumequations(N - 2)
	delta_r = 1 / (N - 1)
	time_index = t2index(t)
	for i = 2:N - 1
		r_i = (i - 1) * delta_r
		r_i_p_half = r_i + 0.5 * delta_r
		r_i_m_half = r_i - 0.5 * delta_r
		j = i - 1#shift the indexing, because our h starts at h[2] from the paper and we omit h[1] and h[N]
		if i == 2
			#the equation for i = 2 has h[i - 1] = h[1] = h[2] (equation 16), so h[i] - h[i - 1] = 0, so that term goes away
			NonlinearEquations.addterm(i - 1, (r_i_p_half * 0.5 * (h[j] + h[j + 1]) * (h[j + 1] - h[j]) / delta_r) / (r_i * delta_r) + fvals[time_index])
		elseif i == N - 1
			#the equation for i = N - 1 has h[i + 1] = h[N] = (h[N - 1] + delta_r * Bi * h_tr(t)) / (1 + delta_r * Bi) -- from equation 18
			NonlinearEquations.addterm(i - 1, ((r_i_p_half * 0.5 * (h[j] + ((h[j] + delta_r * Bi * h_tr[time_index]) / (1 + delta_r * Bi))) * (((h[j] + delta_r * Bi * h_tr[time_index]) / (1 + delta_r * Bi)) - h[j]) / delta_r) - (r_i_m_half * 0.5 * (h[j] + h[j - 1]) * (h[j] - h[j - 1]) / delta_r)) / (r_i * delta_r) + fvals[time_index])
		else
			NonlinearEquations.addterm(i - 1, ((r_i_p_half * 0.5 * (h[j] + h[j + 1]) * (h[j + 1] - h[j]) / delta_r) - (r_i_m_half * 0.5 * (h[j] + h[j - 1]) * (h[j] - h[j - 1]) / delta_r)) / (r_i * delta_r) + fvals[time_index])
		end
	end
end

N = 1001
@assert mod(N - 1, 1000) == 0
R = 10.0
D = 0.5
Kr = 0.5
kappa = 0.1
numtimesteps = 1000
@assert mod(numtimesteps, 1000) == 0
Sy = 0.3
h_tr1 = map(x->1.0-0.7/div(3 * numtimesteps, 10)*x, 1:(div(3 * numtimesteps, 10)))
h_tr2 = map(x->0.3*(1+sin(0.01*x / div(numtimesteps, 1000))),0:(div(7 * numtimesteps, 10) - 1))
h_tr = vcat(h_tr1, h_tr2)
ts = collect(range(0, 100; length=numtimesteps + 1)) * Kr * D / (Sy * R ^ 2)
fvals = zeros(length(ts))
fvals[500 * div(numtimesteps, 1000):600 * div(numtimesteps, 1000)] .= 2e-3
fvals .*= R ^ 2 / (Kr * D ^ 2)
function t2index(t)
	if t > maximum(ts) || t < 0
		error("crazy time: $t")
	else
		return round(Int, t / maximum(ts) * (length(ts) - 1))
	end
end
Bi = kappa * R / Kr

function unpack(p)
	return p
end

function f(u, p, t)
	fvals = unpack(p)
	return icewedge_residuals(u, fvals, h_tr, Bi, t, N, t2index)
end

function f_u(u, p, t)
	fvals = unpack(p)
	return icewedge_h(u, fvals, h_tr, Bi, t, N, t2index)
end

function f_p(u, p, t)
	fvals = unpack(p)
	return icewedge_fvals(u, fvals, h_tr, Bi, t, N, t2index)
end

f_t(u, p, t) = zeros(length(u))

p = fvals
h0 = ones(N - 2)
print("forward run time:")
@time h = DifferentiableBackwardEuler.steps(h0, f, f_u, f_p, f_t, p, ts; ftol=1e-12, method=:newton) * D#multiply by D to convert from h* to h
h = h[div(N - 1, 1000):div(N - 1, 1000):end, 1:div(numtimesteps, 1000):end]
h_vitaly = D .- DelimitedFiles.readdlm("vitaly_solution2.dlm")

function plot(h, i; savefig=true, filename="result.png")
	fig, ax = PyPlot.subplots()
	rs = collect(range(0, R; length=1001))
	ax.plot(rs[2:end - 1], h[:, i], "b", alpha=0.5, label="dan")
	ax.plot(rs[2:end - 1], h_vitaly[2:1000, i], "g", alpha=0.5, label="vitaly")
	ax.plot([rs[end]], [h_tr[i] * D], "k.", label="h_t")
	ax.set_xlabel("r")
	ax.set_xlabel("h")
	ax.set_ylim(-0.1, 0.6)
	ax.legend(loc=3)
	if savefig
		fig.savefig(filename)
	else
		display(fig)
		println()
	end
	PyPlot.close(fig)
end
for (i, j) = enumerate(1:10:size(h, 2) - 1)
	plot(h, j; filename="figs/frame_$(lpad(i, 4, '0')).png")
end
run(`ffmpeg -y -r 5 -f image2 -s 1920x1080 -i figs/frame_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p movie.mp4`)

#=
g(p) = DifferentiableBackwardEuler.steps(h0, f, f_u, f_p, f_t, p, ts; ftol=1e-12, method=:newton)[1, end] * D
print("gradient time: ")
@time grad_zygote = Zygote.gradient(g, p)[1]

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
checkgradientquickly(g, p, grad_zygote, 2)
function checkresiduals()
	fig, ax = PyPlot.subplots()
	residuals = zeros(length(ts) - 1)
	residuals_vitaly = zeros(length(ts) - 1)
	for i = 2:1001
		deltah = ts[i] - ts[i - 1]
		residuals[i - 1] = sum((h[:, i] - h[:, i - 1] - deltah * f(h[:, i], p, ts[i])) .^ 2)
		residuals_vitaly[i - 1] = sum((h_vitaly[2:end - 1, i] - h_vitaly[2:end - 1, i - 1] - deltah * f(h_vitaly[2:end - 1, i], p, ts[i])) .^ 2)
	end
	i_worst = findmax(abs.(residuals .- residuals_vitaly))[2]
	#@show ts[i_worst], fvals[i_worst]
	#@show ts[i_worst - 1], fvals[i_worst - 1]
	ax.plot(residuals, alpha=0.5, label="dan")
	ax.plot(residuals_vitaly, alpha=0.5, label="vitaly")
	ax.legend()
	ax.set_xlabel("iteration number")
	ax.set_ylabel("sum of squared residuals")
	display(fig)
	println()
	PyPlot.close(fig)
end
if N == 1001#this only works if N = 1001
	checkresiduals()
end
=#
