#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
using Flux
using Flux.Tracker
using Flux.Tracker: data, TrackedArray, track, @grad

@time include("utilities.jl")
@time include("inputdeck.jl")
@time include("ex.jl")

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
plotgradient(data(grads[pKs]), obsnode)
