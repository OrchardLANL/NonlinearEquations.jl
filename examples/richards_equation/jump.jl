#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
import Ipopt
using JuMP
import MathOptInterface

include("utilities.jl")
include("inputdeck.jl")
include("ex.jl")

function req_jump(Ks, neighbors::Array, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
	neighborsdict = Dict{Int, Set{Int}}()
	neighbor2neighborindex = Dict{Pair{Int, Int}, Int}()
	for i = 1:size(coords, 2)
		neighborsdict[i] = Set{Int}()
	end
	for (i, (node1, node2)) in enumerate(neighbors)
		push!(neighborsdict[node1], node2)
		push!(neighborsdict[node2], node1)
		neighbor2neighborindex[node1=>node2] = i
		neighbor2neighborindex[node2=>node1] = i
	end
	req_jump(Ks, neighborsdict, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, neighbor2neighborindex)
end

function req_jump(Ks, neighbors_dict::Dict, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs, n2ni)
	model = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=5 * 60.0))
	psi0 = -ones(size(coords, 2))
	psi0[dirichletnodes] = zeros(length(dirichletnodes))
	@variable(model, psi[j=1:size(coords, 2)], start=psi0[j])
	@constraint(model, psi[dirichletnodes] .== dirichletpsis[dirichletnodes])
	dirichletnodesset = Set(dirichletnodes)
	for i = filter(x->!(x in dirichletnodesset), 1:size(coords, 2))
		fixedneighbors = Int[]
		freeneighbors = Int[]
		for n in neighbors_dict[i]
			if n in dirichletnodesset
				push!(fixedneighbors, n)
			else
				push!(freeneighbors, n)
			end
		end
		j = freeneighbors[1]
		@NLconstraint(model, 0 == Qs[i]
					  + sum(2 / (1 / ifelse(psi[i] < 0, (1 - abs(alphas[n2ni[i => nbor]] * psi[i]) ^ (Ns[n2ni[i => nbor]] - 1) * (1 + abs(alphas[n2ni[i => nbor]] * psi[i]) ^ Ns[n2ni[i => nbor]]) ^ -((Ns[n2ni[i => nbor]] - 1) / Ns[n2ni[i => nbor]])) ^ 2 / (1 + abs(alphas[n2ni[i => nbor]] * psi[i]) ^ Ns[n2ni[i => nbor]]) ^ ((Ns[n2ni[i => nbor]] - 1) / (2 * Ns[n2ni[i => nbor]])), 1.0) + 1 / (ifelse(psi[nbor] < 0, (1 - abs(alphas[n2ni[i => nbor]] * psi[nbor]) ^ (Ns[n2ni[i => nbor]] - 1) * (1 + abs(alphas[n2ni[i => nbor]] * psi[nbor]) ^ Ns[n2ni[i => nbor]]) ^ -((Ns[n2ni[i => nbor]] - 1) / Ns[n2ni[i => nbor]])) ^ 2 / (1 + abs(alphas[n2ni[i => nbor]] * psi[nbor]) ^ Ns[n2ni[i => nbor]]) ^ ((Ns[n2ni[i => nbor]] - 1) / (2 * Ns[n2ni[i => nbor]])), 1.0))) * Ks[n2ni[i=>nbor]] * (psi[nbor] + coords[2, nbor] - psi[i] - coords[2, i]) * areasoverlengths[n2ni[i=>nbor]] for nbor in freeneighbors)
					  + sum(2 / (1 / ifelse(psi[i] < 0, (1 - abs(alphas[n2ni[i => nbor]] * psi[i]) ^ (Ns[n2ni[i => nbor]] - 1) * (1 + abs(alphas[n2ni[i => nbor]] * psi[i]) ^ Ns[n2ni[i => nbor]]) ^ -((Ns[n2ni[i => nbor]] - 1) / Ns[n2ni[i => nbor]])) ^ 2 / (1 + abs(alphas[n2ni[i => nbor]] * psi[i]) ^ Ns[n2ni[i => nbor]]) ^ ((Ns[n2ni[i => nbor]] - 1) / (2 * Ns[n2ni[i => nbor]])), 1.0) + 1 / (ifelse(dirichletpsis[nbor] < 0, (1 - abs(alphas[n2ni[i => nbor]] * dirichletpsis[nbor]) ^ (Ns[n2ni[i => nbor]] - 1) * (1 + abs(alphas[n2ni[i => nbor]] * dirichletpsis[nbor]) ^ Ns[n2ni[i => nbor]]) ^ -((Ns[n2ni[i => nbor]] - 1) / Ns[n2ni[i => nbor]])) ^ 2 / (1 + abs(alphas[n2ni[i => nbor]] * dirichletpsis[nbor]) ^ Ns[n2ni[i => nbor]]) ^ ((Ns[n2ni[i => nbor]] - 1) / (2 * Ns[n2ni[i => nbor]])), 1.0))) * Ks[n2ni[i=>nbor]] * (dirichletpsis[nbor] + coords[2, nbor] - psi[i] - coords[2, i]) * areasoverlengths[n2ni[i=>nbor]] for nbor in fixedneighbors))
	end
	@objective(model, Min, 0)
	JuMP.optimize!(model)
	return JuMP.value.(psi), model
end

psi_jump, model = req_jump(Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
res(psi) = req_residuals(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)
jac(psi) = req_psi(psi, Ks, neighbors, areasoverlengths, dirichletnodes, dirichletpsis, coords, alphas, Ns, Qs)

callback(psi_jump, res(psi_jump), jac(psi_jump), 0)
