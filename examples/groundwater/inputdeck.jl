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
	ρ = 997.0#kg/m^3
	g = 9.81#m/s^2
	α = 1e-8#Pa^-1 = s^2*m/kg -- pulled out of thin air, but wanting a contribution where the solid is more compressible than the water
	porosity = 0.25#unitless
	β = 45.8e-11#Pa^-1 = s^2*m/kg -- compressiblity of water is taken from http://hyperphysics.phy-astr.gsu.edu/hbase/Tables/compress.html
	specificstorage = fill(ρ * g * (α + porosity * β), size(coords, 2))#ρg(α+nβ) where ρ is fluid density, g, is gravitational constant, α is aquifer compressiblity, n is porosity, β is fluid compressibility
	return coords, neighbors, areasoverlengths, specificstorage, volumes
end

mins = [0, 0]
maxs = [100, 60]
#ns = [13, 7]
ns = [25, 15]
#ns = [51, 31]
#ns = [75, 45]
#ns = [101, 61]
#ns = [201, 121]
#ns = [401, 241]
#ns = [801, 481]
coords, neighbors, areasoverlengths, specificstorage, volumes = regulargrid(mins, maxs, ns, 1.0)
params = Dict(:clay=>(1.58e-4, 0.244, 1.09, 0.178947368), :claysilt=>(1e-2, 0.488, 1.37, 0.073913043))#(K, alpha, N, sr)
inclay(x, z) = x < 65 && z > 25 && z < 30
function setupparams(coords, neighbors)
	global params
	Ks = zeros(length(neighbors))
	#name	Kh	Kz	alpha	N	sr
	#claysilt	1.00E-02	1.00E-02	0.488	1.37	0.073913043
	#clay	1.58E-04	1.58E-04	0.244	1.09	0.178947368
	for (i, (node1, node2)) in enumerate(neighbors)
		x, z = 0.5 * (coords[:, node1] + coords[:, node2])
		if inclay(x, z)
			Ks[i], _, _ = params[:clay]
		else
			Ks[i], _, _ = params[:claysilt]
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
	dirichleths = zeros(size(coords, 2))
	return Ks, dirichletnodes, dirichleths, Qs
end

Ks, dirichletnodes, dirichleths, Qs = setupparams(coords, neighbors)
