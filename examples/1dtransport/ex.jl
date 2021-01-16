import NonlinearEquations
using OrdinaryDiffEq
import PyPlot
import SparseArrays

#solve u_t + c * u_x = 0 on a periodic 1d domain
@NonlinearEquations.equations exclude=(c,) function t1d(u, c)
	dx = 1 / length(u)
	NonlinearEquations.setnumequations(length(u))
	if c > 0#use an upwind discretization of u_x
		NonlinearEquations.addterm(1, -c * (u[1] - u[end]) / dx)
		for i = 2:length(u)
			NonlinearEquations.addterm(i, -c * (u[i] - u[i - 1]) / dx)
		end
	else
		for i = 1:length(u) - 1
			NonlinearEquations.addterm(i, -c * (u[i + 1] - u[i]) / dx)
		end
		NonlinearEquations.addterm(length(u), -c * (u[1] - u[end]) / dx)
	end
end

for c in [exp(1), -pi]
	n = 1000
	p = nothing
	tspan = [0.0, 1 / abs(c)]
	u0 = map(i->i < n / 8 || i > 7 * n / 8 ? 1.0 : 0.0, 1:n) + map(x->exp(-(x - 0.5) ^ 2 * 64), range(0, 1; length=n))
	f(u, p, t) = t1d_residuals(u, c)
	f_u(u, p, t) = t1d_u(u, c)
	odef = ODEFunction(f; jac=f_u, jac_prototype=f_u(u0, p, 0.0))
	prob = ODEProblem(odef, u0, tspan, p)
	@time u_explicit = solve(prob, Euler(); dt=0.1 / (abs(c) * n), u0=prob.u0, p=prob.p)
	@time u_implicit = solve(prob, ImplicitEuler(); u0=prob.u0, p=prob.p, abstol=1e-4, reltol=1e-4)

	fig, ax = PyPlot.subplots()
	ax.plot(range(0, 1; length=length(u0)), u0, label="init")
	ax.plot(range(0, 1; length=length(u0)), u_explicit(tspan[end]), label="explicit")
	ax.plot(range(0, 1; length=length(u0)), u_implicit(tspan[end]), label="implicit")
	ax.legend()
	display(fig)
	PyPlot.println()
	PyPlot.close(fig)
end
