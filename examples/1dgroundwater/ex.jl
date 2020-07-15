using Test
using DiffEqSensitivity
using OrdinaryDiffEq
import DifferentiableBackwardEuler
import NonlinearEquations
import Random
import SparseArrays
import Zygote

import NonlinearEquations
import SparseArrays
@NonlinearEquations.equations function gw1d(h, Qs)
	NonlinearEquations.setnumequations(length(h))
	NonlinearEquations.addterm(1, -2 * h[1] + h[2] + Qs[1])
	for i = 2:length(h) - 1
		NonlinearEquations.addterm(i, h[i - 1] - 2 * h[i] + h[i + 1] + Qs[i])
	end
	NonlinearEquations.addterm(length(h), -2 * h[end] + h[end - 1] + Qs[end])
end

function solveforh(Qs)
	res(h) = gw1d_residuals(h, Qs)
	jac(h) = gw1d_h(h, Qs)
	h0 = zeros(N)
	h = jac(h0) \ -res(h0)
	return h
end

N = 25
Qs = 8 * ones(N) / N ^ 2
h = solveforh(Qs)

#compute the gradient for steady state
Random.seed!(1)
obsnode = rand(1:length(h))
g(h, Qs) = h[obsnode]
function g_h(h, p)
	retval = zeros(length(h))
	retval[obsnode] = 1.0
	return retval
end
g_Qs(h, Qs) = zeros(length(Qs))
f_h(h, Qs) = gw1d_h(h, Qs)
f_Qs(h, Qs) = gw1d_Qs(h, Qs)
grad = NonlinearEquations.gradient(h, Qs, g_h, g_Qs, f_h, f_Qs)
t1 = @elapsed grad = NonlinearEquations.gradient(h, Qs, g_h, g_Qs, f_h, f_Qs)

fdgrad = similar(grad)
sortedgradindices = sort(1:length(grad); by=i->abs(grad[i]), rev=true)
t2 = @elapsed for i = 1:length(fdgrad)
	global fdgrad
	global Qs
	global h
	dk = 1e-8
	h0 = h
	Qs0 = copy(Qs)
	thisQs = copy(Qs)
	thisQs[i] += dk
	thish = solveforh(thisQs)
	fdgrad[i] = (thish[obsnode] - h0[obsnode]) / dk
end
@test isapprox(grad, fdgrad; rtol=1e-4)#test the steady state gradients are working

u0 = zeros(N)
p = Qs
tspan = [0.0, 1e4]
f(u, p, t) = gw1d_residuals(u, p)
f_u(u, p, t) = gw1d_h(u, p)
f_u(storage, u, p, t) = gw1d_h!(storage, u, p)
f_p(u, p, t) = gw1d_Qs(u, p)
f_p(storage, u, p, t) = gw1d_Qs!(storage, u, p)
odef = ODEFunction(f; jac=f_u, jac_prototype=f_u(u0, p, 0.0), paramjac=f_p)
prob = ODEProblem(odef, u0, tspan, p)
soln_diffeq = solve(prob, ImplicitEuler(); u0=prob.u0, p=prob.p, abstol=1e-8, reltol=1e-8)
soln_dbe = DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, (args...)->zeros(length(u0)), p, soln_diffeq.t; ftol=1e-12)
@test isapprox(soln_diffeq[:, :], soln_dbe)#test DifferentiableBackwardEuler is working for the forward solve
loss(p) = DifferentiableBackwardEuler.steps(u0, f, f_u, f_p, (args...)->zeros(length(u0)), p, soln_diffeq.t; ftol=1e-12)[obsnode, end]
dgdp_zygote = Zygote.gradient(loss, p)[1]
@test isapprox(dgdp_zygote, grad)#test that the steady state derivative matches the equivalent derivative obtained with the transient solution
