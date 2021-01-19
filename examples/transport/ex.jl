using Test
import DifferentiableBackwardEuler
import DPFEHM
import NonlinearEquations
import PyPlot
import SparseArrays
import Zygote

#solve u_t + div(v * u) - div(D * grad(u)) - Q = 0 on an unstructured mesh using an upwind discretization of the velocity term (or downwind if use_upwind=false)
@NonlinearEquations.equations exclude=(areasoverlengths, volumes, dirichletnodes, coords) function transport(u, vxs, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords; use_upwind=true)
	@assert size(coords, 1) == 3#it is assumed that this is a 3d problem
	NonlinearEquations.setnumequations(length(u))
	for i = 1:length(u)
		if i in dirichletnodes
			NonlinearEquations.addterm(i, u[i] - dirichletus[i])
		else
			NonlinearEquations.addterm(i, Qs[i] / (volumes[i]))
		end
	end
	delta = zeros(eltype(coords), size(coords, 1))
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			#add the advection term
			delta .= coords[:, node1] .- coords[:, node2]
			is_upwind = ((vxs[i] * delta[1] + vys[i] * delta[2] + vzs[i] * delta[3]) > 0)
			if (is_upwind && use_upwind) || (!is_upwind && !use_upwind)#!use_upwind basically means use_downwind and !is_upwind basically means is_downwind
				if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
					NonlinearEquations.addterm(node1, -(vxs[i] * delta[1] + vys[i] * delta[2] + vzs[i] * delta[3]) * (u[node1] - u[node2]) * areasoverlengths[i] / (volumes[node1]))
				elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
					NonlinearEquations.addterm(node1, -(vxs[i] * delta[1] + vys[i] * delta[2] + vzs[i] * delta[3]) * (u[node1] - dirichletus[node2]) * areasoverlengths[i] / (volumes[node1]))
				end
			end
			#add the diffusion term
			if !(node1 in dirichletnodes) && !(node2 in dirichletnodes)
				NonlinearEquations.addterm(node1, Ds[i] * (u[node2] - u[node1]) * areasoverlengths[i] / (volumes[node1]))
			elseif !(node1 in dirichletnodes) && node2 in dirichletnodes
				NonlinearEquations.addterm(node1, Ds[i] * (dirichletus[node2] - u[node1]) * areasoverlengths[i] / (volumes[node1]))
			end
		end
	end
end

mins = [0, 0]; maxs = [3, 2]#size of the domain, in meters
ns = [30, 20]#number of nodes on the grid
coords, neighbors, areasoverlengths, volumes = DPFEHM.regulargrid2d(mins, maxs, ns, 1.0)#build the grid
coords = vcat(coords, zeros(size(coords, 2))')#turn it into a 3d coords

c = 1.0
tspan = [0.0, 1.0 / abs(c)]
u0 = map(i->coords[1, i] < 1 / 2  && coords[1, i] > 0 && coords[2, i] > 0.25 && coords[2, i] < 0.75 ? 1.0 : 0.0, 1:size(coords, 2)) + map(i->exp(-((coords[1, i] - 1.0) ^ 2 + (coords[2, i] - 1.0) ^ 2) * 64), 1:size(coords, 2))
vxs = map(neighbor->c, neighbors)
vys = map(neighbor->c * 0.5, neighbors)
vzs = zeros(length(neighbors))
Ds = 1e-2 * ones(length(neighbors))
Qs = zeros(size(coords, 2))
dirichletnodes = [collect(1:ns[2]); collect(size(coords, 2) - ns[2] + 1:size(coords, 2))]
dirichletus = zeros(size(coords, 2))
f(u, p, t) = transport_residuals(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_u(u, p, t) = transport_u(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_p(u, p, t) = transport_vxs(u, p, vys, vzs, Ds, neighbors, areasoverlengths, dirichletnodes, dirichletus, Qs, volumes, coords)
f_t(u, p, t) = zeros(length(u))
function solveforu(vxs)
	DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, f_t, vxs, tspan[1], tspan[2]; abstol=1e-4, reltol=1e-4)
end
@time u_implicit = solveforu(vxs)

fig, axs = PyPlot.subplots(2, 1; figsize=(8, 8))
img = axs[1].imshow(reshape(u0, reverse(ns)...), extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower")
fig.colorbar(img, ax=axs[1])
img = axs[2].imshow(reshape(u_implicit[:, end], reverse(ns)...), extent=[mins[1], maxs[1], mins[2], maxs[2]], origin="lower")
fig.colorbar(img, ax=axs[2])
fig.tight_layout()
display(fig)
PyPlot.println()
PyPlot.close(fig)

_, gradient_node = findmin(map(i->sum((coords[:, i] .- [1.5, 1.0, 0.0]) .^ 2), 1:size(coords, 2)))
g = x->solveforu(x)[gradient_node, end]
@time gradg = Zygote.gradient(g, vxs)[1]
function checkgradientquickly(f, x0, gradf, n; delta::Float64=1e-8, kwargs...)
	indicestocheck = sort(collect(1:length(x0)), by=i->abs(gradf[i]), rev=true)[1:n]
	f0 = f(x0)
	for i in indicestocheck
		x = copy(x0)
		x[i] += delta
		fval = f(x)
		grad_f_i = (fval - f0) / delta
		@test isapprox(gradf[i], grad_f_i; kwargs...)
	end
end
checkgradientquickly(g, vxs, gradg, 3; atol=1e-4, rtol=1e-3)
nothing
