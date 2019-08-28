#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
using Test
using Flux
using Flux.Tracker
using Flux.Tracker: data, TrackedArray, track, @grad

include("utilities.jl")
include("inputdeck.jl")
include("ex.jl")

solveforpsi(Ks::TrackedArray) = track(solveforpsi, Ks)

@grad function solveforpsi(Ks)
	psi = solveforpsi(data(Ks))
	back = delta->begin
		lambda = transpose(f_psi(psi, data(Ks))) \ delta
		return (-transpose(f_Ks(psi, data(Ks))) * lambda, )
	end
	return psi, back
end

pKs = param(Ks)
predict() = solveforpsi(pKs)
loss() = predict()[obsnode]
theta = Params([pKs])
grads = Tracker.gradient(() -> loss(), theta)
@test data(grads[pKs]) ≈ grad

pKs = param(Ks)
predict() = solveforpsi(pKs)
loss() = predict()[obsnode] ^ 2
theta = Params([pKs])
grads = Tracker.gradient(() -> loss(), theta)
@test data(grads[pKs]) ≈ 2 * psi[obsnode] * grad

pKs = param(Ks)
predict() = solveforpsi(exp.(log.(pKs)))
loss() = predict()[obsnode]
theta = Params([pKs])
grads = Tracker.gradient(() -> loss(), theta)
@test data(grads[pKs]) ≈ grad

logKs = log.(Ks)
plogKs = param(logKs)
predict() = solveforpsi(exp.(plogKs))
loss() = predict()[obsnode]
theta = Params([plogKs])
grads = Tracker.gradient(() -> loss(), theta)
@test data(grads[plogKs]) ≈ grad .* Ks

logKs = log.(Ks)
plogKs = param(logKs)
predict() = solveforpsi(exp.(plogKs))
loss() = predict()[obsnode] ^ 2
theta = Params([plogKs])
grads = Tracker.gradient(() -> loss(), theta)
@test data(grads[plogKs]) ≈ 2 * psi[obsnode] * grad .* Ks
