import LinearAlgebra
import NonlinearEquations
import PyPlot
import SparseArrays

function analytical(x, t)
	if x == 0
		return 1.0
	elseif x > ((0.5 + 1 / sqrt(2)) * t)
		return 0.0
	else
		return 0.5 * (sqrt(-2 + t * (sqrt(4 * x / t + 1) - 1) / x + 1) + 1)
	end
end

function getfreenodes(n, dirichletnodes)
	isfreenode = fill(true, n)
	isfreenode[dirichletnodes] .= false
	nodei2freenodei = fill(-1, length(isfreenode))
	freenodei2nodei = Array{Int}(undef, sum(isfreenode))
	j = 1
	for i = 1:length(isfreenode)
		if isfreenode[i]
			nodei2freenodei[i] = j
			freenodei2nodei[j] = i
			j += 1
		end
	end
	return isfreenode, nodei2freenodei, freenodei2nodei
end

f(S) = S ^ 2 / (S ^ 2 + (1 - S) ^ 2)
Ss = 0:1e-2:1
fig, ax = PyPlot.subplots()
ax.plot(Ss, f.(Ss))
ax.set_xlabel("S")
ax.set_ylabel("Fractional Flow")
display(fig)
println()
PyPlot.close(fig)

@NonlinearEquations.equations function bl(Sfree, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei)
	numfreenodes = sum(isfreenode)
	NonlinearEquations.setnumequations(numfreenodes)
	for (i, (node_a, node_b)) in enumerate(neighbors)
		for (node1, node2) in [(node_a, node_b), (node_b, node_a)]
			if isfreenode[node1] && isfreenode[node2]
				j1 = nodei2freenodei[node1]
				j2 = nodei2freenodei[node2]
				if p[node2] > p[node1]
					NonlinearEquations.addterm(j1, Ks[i] * ((p[node2] - p[node1]) / (xs[node2] - xs[node1])) * (Sfree[j2] ^ 2 / (Sfree[j2] ^ 2 + (1 - Sfree[j2]) ^ 2) - Sfree[j1] ^ 2 / (Sfree[j1] ^ 2 + (1 - Sfree[j1]) ^ 2)) * area / (xs[node2] - xs[node1]))
				end
			elseif isfreenode[node1] && !isfreenode[node2]
				j1 = nodei2freenodei[node1]
				if p[node2] > p[node1]
					NonlinearEquations.addterm(j1, Ks[i] * ((p[node2] - p[node1]) / (xs[node2] - xs[node1])) * (dirichletS[node2] ^ 2 / (dirichletS[node2] ^ 2 + (1 - dirichletS[node2]) ^ 2) - Sfree[j1] ^ 2 / (Sfree[j1] ^ 2 + (1 - Sfree[j1]) ^ 2)) * area / (xs[node2] - xs[node1]))
				end
			end
		end
	end
end

function bl_simple(S, f=S->S^2/(S^2+(1-S)^2), area=1)
	n = length(S)
	dx = 1 / n
	dS = zeros(n)
	dS[1] = (f(1) - f(S[1])) * area / dx
	for i = 2:length(S)
		dS[i] = (f(S[i - 1]) - f(S[i])) * area / dx
	end
	return dS
end

n = 1000
neighbors = [i=>i + 1 for i = 1:n - 1]
Ks = ones(length(neighbors))
xs = 0:1/(n - 1):1
S = zeros(n)
dirichletnodes = [1]
dirichletS = zeros(length(S))
dirichletS[1] = 1.0
S[dirichletnodes] = dirichletS[dirichletnodes]
isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(S), dirichletnodes)
Sfree = S[isfreenode]
Sfree_implicit = S[isfreenode]
dt = 0.5 / n
finalt = 0.5
ts = 0:dt:finalt
analytical_solution = map(x->analytical(x, finalt), xs)
p = map(x->1 - x, xs)
area = 1
for t in ts
	global r1, r2
	r1 = bl_simple(Sfree)
	r2 = bl_residuals(Sfree, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei)
	if !(r1 ≈ r2)
		@show t, sum((r1 .- r2) .^ 2)
		error("blah")
	end
	Sfree .+= dt * r2
	if any(isnan.(Sfree))
		@show t
		error("nan")
	end
	Sfree_implicit .+= (LinearAlgebra.I - dt * bl_Sfree(Sfree_implicit, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei)) \ (dt * bl_residuals(Sfree_implicit, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei))
end
S[isfreenode] = Sfree
S_implicit = copy(S)
S_implicit[isfreenode] = Sfree_implicit
fig, ax = PyPlot.subplots()
ax.plot(xs, S, label="explicit")
ax.plot(xs, S_implicit, label="implicit")
ax.plot(xs, analytical_solution, label="analytical")
ax.legend()
ax.set_xlabel("x")
ax.set_xlabel("S")
fig.tight_layout()
display(fig)
println()
PyPlot.close(fig)

n = 1000
neighbors = [i=>i + 1 for i = 1:n - 1]
Ks = ones(length(neighbors))
xs = 0:1/(n - 1):1
S = zeros(n)
dirichletnodes = [length(S)]
dirichletS = zeros(length(S))
dirichletS[end] = 1.0
S[dirichletnodes] = dirichletS[dirichletnodes]
isfreenode, nodei2freenodei, freenodei2nodei = getfreenodes(length(S), dirichletnodes)
Sfree = S[isfreenode]
Sfree_implicit = S[isfreenode]
dt = 0.5 / n
finalt = 0.5
ts = 0:dt:finalt
analytical_solution = map(x->analytical(1 - x, finalt), xs)
p = map(x->x, xs)
area = 1
for t in ts
	r2 = bl_residuals(Sfree, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei)
	Sfree .+= dt * r2
	if any(isnan.(Sfree))
		@show t
		error("nan")
	end
	Sfree_implicit .+= (LinearAlgebra.I - dt * bl_Sfree(Sfree_implicit, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei)) \ (dt * bl_residuals(Sfree_implicit, p, Ks, neighbors, dirichletnodes, dirichletS, area, xs, isfreenode, freenodei2nodei, nodei2freenodei))
end
S[isfreenode] = Sfree
S_implicit = copy(S)
S_implicit[isfreenode] = Sfree_implicit
fig, ax = PyPlot.subplots()
ax.plot(xs, S, label="explicit")
ax.plot(xs, S_implicit, label="implicit")
ax.plot(xs, analytical_solution, label="analytical")
ax.legend()
ax.set_xlabel("x")
ax.set_xlabel("S")
fig.tight_layout()
display(fig)
println()
PyPlot.close(fig)
