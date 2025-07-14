using Test
import Calculus
import ChainRulesCore
import DSP
import NonlinearEquations
import Random
import SparseArrays

Random.seed!(0)

exorig = :(p[1] * x ^ 2 + p[2] * x + p[3])
ex, d = NonlinearEquations.replacerefswithsyms(exorig)
exold = NonlinearEquations.replacesymswithrefs(ex, d)
@test exorig == exold

ex = :(p[1] * x[1] ^ 2 + p[2] * x[2 * y + f(z)] + p[3])
diffs, refs = NonlinearEquations.differentiatewithrefs(ex, :x)
for i = 1:2
	if refs[i] == [:(2 * y + f(z))]
		@test Calculus.simplify(diffs[i]) == :(p[2])
	elseif refs[i] == [:(1)]
		@test Calculus.simplify(diffs[i]) == :(p[1] * (2 * x[1]))
	else
		error("the refs are bad")
	end
end

#formulate a quadratic equation
@NonlinearEquations.equations function quadeq(x, p; asdf=1)
	NonlinearEquations.setnumequations(1)
	NonlinearEquations.addterm(1, p[1] * x[1] ^ 2 + p[2] * x[1] + p[3])
end

#solve (x-1)(x+1)
p = [1.0, 0.0, -1.0]
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_x(x, p), [0.01])
@test root[1] ≈ 1.0
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_x(x, p), [100.0])
@test root[1] ≈ 1.0
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_x(x, p), [-0.01])
@test root[1] ≈ -1.0
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_x(x, p), [-23])
@test root[1] ≈ -1.0
#test the gradient
g(x, p) = x[1] * (p[1] + 2 * p[2] + 3 * p[3])
g_x(x, p) = [(p[1] + 2 * p[2] + 3 * p[3])]
g_p(x, p) = x[1] * [1, 2, 3]
f_x(x, p) = quadeq_x(x, p)
f_p(x, p) = quadeq_p(x, p)
g(p) = (-p[2] - sqrt(p[2] ^ 2 - 4 * p[1] * p[3])) / (2 * p[1]) * (p[1] + 2 * p[2] + 3 * p[3])
analytical_gradient = Calculus.gradient(g)
@test analytical_gradient(p) ≈ NonlinearEquations.gradient(root, p, g_x, g_p, f_x, f_p)

#solve (2x-3)(7x+5)
p = [14.0, -11.0, -15]
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_x(x, p), [1.0])
@test root[1] ≈ 3 / 2
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_x(x, p), [0.0])
@test root[1] ≈ -5 / 7

#formulate a quadratic equation in a different way
@NonlinearEquations.equations function quadeq2(x::T, p; asdf=1) where {T}
	NonlinearEquations.setnumequations(1)
	NonlinearEquations.addterm(1, p[1] * x[1] ^ 2)
	NonlinearEquations.addterm(1, p[2] * x[1])
	NonlinearEquations.addterm(1, p[3])
end

#solve (x-1)(x+1)
p = [1.0, 0.0, -1.0]
root = NonlinearEquations.newton(x->quadeq2_residuals(x, p), x->quadeq2_x(x, p), [0.01])
@test root[1] ≈ 1.0
root = NonlinearEquations.newton(x->quadeq2_residuals(x, p), x->quadeq2_x(x, p), [100.0])
@test root[1] ≈ 1.0
root = NonlinearEquations.newton(x->quadeq2_residuals(x, p), x->quadeq2_x(x, p), [-0.01])
@test root[1] ≈ -1.0
root = NonlinearEquations.newton(x->quadeq2_residuals(x, p), x->quadeq2_x(x, p), [-23])
@test root[1] ≈ -1.0

#solve (2x-3)(7x+5)
p = [14.0, -11.0, -15]
root = NonlinearEquations.newton(x->quadeq2_residuals(x, p), x->quadeq2_x(x, p), [0.0])
@test root[1] ≈ -5 / 7
root = NonlinearEquations.newton(x->quadeq2_residuals(x, p), x->quadeq2_x(x, p), [1.0])
@test root[1] ≈ 3 / 2

#formulate the steady-state diffusion equation with a heterogeneous diffusion coefficient
@NonlinearEquations.equations function diffusion(x, p, n)
	@assert length(x) == n
	@assert length(p) == n + 1
	NonlinearEquations.setnumequations(n)
	dx2 = 1 / (n + 1) ^ 2
	NonlinearEquations.addterm(1, ((p[1] + p[2]) * x[1] - p[2] * x[2] - p[1]) / dx2)
	for eqnum = 2:n - 1
		NonlinearEquations.addterm(eqnum, (-p[eqnum] * x[eqnum - 1] + (p[eqnum] + p[eqnum + 1]) * x[eqnum] - p[eqnum + 1] * x[eqnum + 1]) / dx2)
	end
	NonlinearEquations.addterm(n, ((p[end - 1] + p[end]) * x[end] - p[end - 1] * x[end - 1]) / dx2)
end

#solve the diffusion equation with head=1 at x=0 and head=0 at x=1 and a homogeneous diffusion coefficient
n = 10 ^ 3
p = ones(n + 1)
head = NonlinearEquations.newton(x->diffusion_residuals(x, p, n), x->diffusion_x(x, p, n), zeros(n); numiters=1)
dx2 = (1 / (n + 1)) ^ 2
otherhead = SparseArrays.spdiagm(-1=>-ones(n - 1) / dx2, 0=>2 * ones(n) / dx2, 1=>-ones(n - 1) / dx2) \ [[1 / dx2]; zeros(n - 1)]
@test head ≈ otherhead

#solve the diffusion equation with head=1 at x=0 and head=0 at x=1 and a heterogeneous diffusion coefficient
n = 10 ^ 3
windowsize = div(n, 10)
p = exp.(DSP.conv(0.1 * rand(n + windowsize + 1), ones(windowsize))[windowsize + 1:end - windowsize + 1])
head = NonlinearEquations.newton(x->diffusion_residuals(x, p, n), x->diffusion_x(x, p, n), zeros(n); numiters=1)
dx2 = (1 / (n + 1)) ^ 2
otherhead = SparseArrays.spdiagm(-1=>-p[2:n] / dx2, 0=>(p[1:n] + p[2:n + 1]) / dx2, 1=>-p[2:n] / dx2) \ [[p[1] / dx2]; zeros(n - 1)]
@test head ≈ otherhead


#formulate the boundary value problem u''(x)=u^2-sin^2(x)-sin(x) with u(0)=0 and u(1)=0, which has u(x)=sin(x) as a solution on the interval [0, π]
@NonlinearEquations.equations function nonlinearbvp(u, p, n)
	NonlinearEquations.setnumequations(n)
	dx2 = (pi / (n + 1)) ^ 2
	xs = range(0, pi; length=n + 2)[2:end - 1]
	NonlinearEquations.addterm(1, -(2 * u[1] - u[2] - 0) / dx2 - u[1] ^ 2 + sin(xs[1]) ^ 2 + sin(xs[1]))
	for eqnum = 2:n - 1
		NonlinearEquations.addterm(eqnum, -(2 * u[eqnum] - u[eqnum - 1] - u[eqnum + 1]) / dx2 - u[eqnum] ^ 2 + sin(xs[eqnum]) ^ 2 + sin(xs[eqnum]))
	end
	NonlinearEquations.addterm(n, -(2 * u[n] - u[n - 1] - 0 * sin(1)) / dx2 - u[n] ^ 2 + sin(xs[n]) ^ 2 + sin(xs[n]))
end
#now solve the boundary value problem
n = 10 ^ 4
p = []
u = NonlinearEquations.newton(x->nonlinearbvp_residuals(x, p, n), x->nonlinearbvp_u(x, p, n), zeros(n); numiters=100)
u_analytical = map(x->sin(x), collect(range(0, pi; length=n + 2))[2:end - 1])
@test u ≈ u_analytical
