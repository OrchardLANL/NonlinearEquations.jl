function regulargrid(mins, maxs, ns, dz)
	linearindex = (i1, i2)->i2 + ns[2] * (i1 - 1)
	coords = Array{Float64}(undef, 2, prod(ns))
	xs = range(mins[1]; stop=maxs[1], length=ns[1])
	ys = range(mins[2]; stop=maxs[2], length=ns[2])
	dx = xs[2] - xs[1]
	dy = ys[2] - ys[1]
	j = 1
	neighbors = Array{Pair{Int, Int}}(undef, 2 * prod(ns) - ns[1] - ns[2])
	areasoverlengths = Array{Float64}(undef, 2 * prod(ns) - ns[1] - ns[2])
	volumes = Array{Float64}(undef, 0)
	for i1 = 1:ns[1]
		for i2 = 1:ns[2]
			push!(volumes, dx * dy * dz)
			coords[1, linearindex(i1, i2)] = xs[i1]
			coords[2, linearindex(i1, i2)] = ys[i2]
			if i1 < ns[1]
				neighbors[j] = linearindex(i1, i2)=>linearindex(i1 + 1, i2)
				areasoverlengths[j] = dy * dz / dx
				j += 1
			end
			if i2 < ns[2]
				neighbors[j] = linearindex(i1, i2)=>linearindex(i1, i2 + 1)
				areasoverlengths[j] = dx * dz / dy
				j += 1
			end
		end
	end
	return coords, neighbors, areasoverlengths, volumes
end

mins = [0, 0]
maxs = [100, 60]
#ns = [51, 31]
#ns = [75, 45]
ns = [101, 61]
#ns = [201, 121]
#ns = [401, 241]
#ns = [801, 481]
coords, neighbors, areasoverlengths, volumes = regulargrid(mins, maxs, ns, 1.0)
params = Dict(:clay=>(1.58e-4, 0.244, 1.09, 0.178947368), :claysilt=>(1e-2, 0.488, 1.37, 0.073913043))#(K, alpha, N, sr)
inclay(x, z) = x < 65 && z > 25 && z < 30
function setupparams(coords, neighbors)
	global params
	Ks = zeros(length(neighbors))
	alphas = zeros(length(neighbors))
	Ns = zeros(length(neighbors))
	#name	Kh	Kz	alpha	N	sr
	#claysilt	1.00E-02	1.00E-02	0.488	1.37	0.073913043
	#clay	1.58E-04	1.58E-04	0.244	1.09	0.178947368
	for (i, (node1, node2)) in enumerate(neighbors)
		x, z = 0.5 * (coords[:, node1] + coords[:, node2])
		if inclay(x, z)
			Ks[i], alphas[i], Ns[i] = params[:clay]
		else
			Ks[i], alphas[i], Ns[i] = params[:claysilt]
		end
	end
	dirichletnodes = Int[]
	Qs = zeros(size(coords, 2))
	for i = 1:size(coords, 2)
		if coords[2, i] == mins[2]
			push!(dirichletnodes, i)
		end
		if coords[2, i] == maxs[2]
			Qs[i] = 0.00055
		end
	end
	dirichletpsis = zeros(size(coords, 2))
	return Ks, dirichletnodes, dirichletpsis, alphas, Ns, Qs
end

Ks, dirichletnodes, dirichletpsis, alphas, Ns, Qs = setupparams(coords, neighbors)
