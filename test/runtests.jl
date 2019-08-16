using Test
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
	if refs[i] == :(2 * y + f(z))
		@test diffs[i] == :(p[2])
	elseif refs[i] == :(1)
		@test diffs[i] == :(p[1] * (2 * x[1]))
	else
		error("the refs are bad")
	end
end

#formulate a quadratic equation
@NonlinearEquations.equations function quadeq(x, p)
	NonlinearEquations.setnumequations(1)
	NonlinearEquations.addterm(1, p[1] * x[1] ^ 2 + p[2] * x[1] + p[3])
end

#solve (x-1)(x+1)
p = [1.0, 0.0, -1.0]
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_jacobian(x, p), [0.01])
@test root[1] ≈ 1.0
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_jacobian(x, p), [100.0])
@test root[1] ≈ 1.0
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_jacobian(x, p), [-0.01])
@test root[1] ≈ -1.0
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_jacobian(x, p), [-23])
@test root[1] ≈ -1.0

#solve (2x-3)(7x+5)
p = [14.0, -11.0, -15]
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_jacobian(x, p), [0.0])
@test root[1] ≈ -5 / 7
root = NonlinearEquations.newton(x->quadeq_residuals(x, p), x->quadeq_jacobian(x, p), [1.0])
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
head = NonlinearEquations.newton(x->diffusion_residuals(x, p, n), x->diffusion_jacobian(x, p, n), zeros(n); numiters=1)
dx2 = (1 / (n + 1)) ^ 2
otherhead = SparseArrays.spdiagm(-1=>-ones(n - 1) / dx2, 0=>2 * ones(n) / dx2, 1=>-ones(n - 1) / dx2) \ [[1 / dx2]; zeros(n - 1)]
@test head ≈ otherhead

#solve the diffusion equation with head=1 at x=0 and head=0 at x=1 and a heterogeneous diffusion coefficient
n = 10 ^ 3
windowsize = div(n, 10)
p = exp.(DSP.conv(0.1 * rand(n + windowsize + 1), ones(windowsize))[windowsize + 1:end - windowsize + 1])
head = NonlinearEquations.newton(x->diffusion_residuals(x, p, n), x->diffusion_jacobian(x, p, n), zeros(n); numiters=1)
dx2 = (1 / (n + 1)) ^ 2
otherhead = SparseArrays.spdiagm(-1=>-p[2:n] / dx2, 0=>(p[1:n] + p[2:n + 1]) / dx2, 1=>-p[2:n] / dx2) \ [[p[1] / dx2]; zeros(n - 1)]
@test head ≈ otherhead