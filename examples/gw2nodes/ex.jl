import SparseArrays
import NonlinearEquations
@NonlinearEquations.equations function gw1d(h, Q)
	NonlinearEquations.setnumequations(3)
	NonlinearEquations.addterm(1, -2 * h[1] + h[2] + Q[1])
	NonlinearEquations.addterm(2, h[1] - 2 * h[2] + h[3] + Q[1])
	NonlinearEquations.addterm(3, -2 * h[3] + h[2] + Q[3])
end
function solveforh(Q)
	res(h) = gw1d_residuals(h, Q)
	jac(h) = gw1d_h(h, Q)
	h0 = zeros(N)
	h = jac(h0) \ -res(h0)
	return h
end

N = 3
Q = 8 * ones(N) / N ^ 2
h = solveforh(Q)
