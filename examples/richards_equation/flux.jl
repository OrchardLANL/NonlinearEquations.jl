#this example is modified from https://numericalenvironmental.wordpress.com/2018/05/26/a-steady-state-variably-saturated-flow-model-in-vertical-cross-section-a-finite-difference-approach-using-julia/
using Test
import Flux
import Zygote

include("utilities.jl")
include("inputdeck.jl")
include("ex.jl")

@Zygote.adjoint function solveforpsi(Ks)
	psi = solveforpsi(Ks)
	back = delta->begin
		lambda = transpose(f_psi(psi, Ks)) \ delta
		return (-transpose(f_Ks(psi, Ks)) * lambda, )
	end
	return psi, back
end

predict() = solveforpsi(Ks)
loss() = predict()[obsnode]
theta = Flux.params([Ks])
grads = Flux.gradient(() -> loss(), theta)
@test grads[Ks] ≈ grad

predict() = solveforpsi(Ks)
loss() = predict()[obsnode] ^ 2
theta = Flux.params([Ks])
grads = Flux.gradient(() -> loss(), theta)
@test grads[Ks] ≈ 2 * psi[obsnode] * grad

predict() = solveforpsi(exp.(log.(Ks)))
loss() = predict()[obsnode]
theta = Flux.params([Ks])
grads = Flux.gradient(() -> loss(), theta)
@test grads[Ks] ≈ grad

logKs = log.(Ks)
predict() = solveforpsi(exp.(logKs))
loss() = predict()[obsnode]
theta = Flux.params([logKs])
grads = Flux.gradient(() -> loss(), theta)
@test grads[logKs] ≈ grad .* Ks

logKs = log.(Ks)
predict() = solveforpsi(exp.(logKs))
loss() = predict()[obsnode] ^ 2
theta = Flux.params([logKs])
grads = Flux.gradient(() -> loss(), theta)
@test grads[logKs] ≈ 2 * psi[obsnode] * grad .* Ks
