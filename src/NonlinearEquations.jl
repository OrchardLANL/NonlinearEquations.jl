module NonlinearEquations

import Calculus
import MacroTools
import SparseArrays

function codegen_addterm_residuals(equationnum, term)
	return quote 
		residuals[$equationnum] += $term
	end
end

function codegen_addterm_jacobian(equationnum, term, xsym)
	derivatives, refs = NonlinearEquations.differentiatewithrefs(term, xsym)
	for ref in refs
		if length(ref) != 1
			error("must be a reference with a single index")
		end
	end
	refs = map(ref->replaceall(ref[1], :end, :(length($xsym))), refs)
	q = quote end
	for (derivative, ref) in zip(derivatives, refs)
		newcode = quote
			push!(I, $equationnum)
			push!(J, $(ref))
			push!(V, $(derivative))
		end
		append!(q.args, newcode.args)
	end
	return q
end

function differentiatewithrefs(exorig, x::Symbol)
	ex, dict = replacerefswithsyms(exorig)
	diffsyms = Symbol[]
	diffrefs = Any[]
	for (junksym, (sym, ref)) in dict
		if sym == x
			push!(diffsyms, junksym)
			push!(diffrefs, ref)
		end
	end
	diffs = Calculus.differentiate(ex, diffsyms)
	diffs = map(diff->Calculus.simplify(replacesymswithrefs(diff, dict)), diffs)
	return diffs, diffrefs
end

macro equations(fundef)
	@MacroTools.capture(fundef, function funsym_(xsym_, psym_, args__) body_ end) || error("unsupported function definition")
	body = macroexpand(Main, body)
	body_residuals = MacroTools.postwalk(x->replacenumequations(x, :(residuals = zeros(numequations))), body)
	body_residuals = MacroTools.postwalk(x->replaceaddterm(x, codegen_addterm_residuals), body_residuals)
	q_residuals = quote
		function $(esc(Symbol(funsym, :_residuals)))($xsym, $psym, $(args...))
			$body_residuals
			return residuals
		end
	end
	body_jacobian = MacroTools.postwalk(x->replacenumequations(x, :()), body)
	body_jacobian = MacroTools.postwalk(x->replaceaddterm(x, (eqnum, term)->codegen_addterm_jacobian(eqnum, term, xsym)), body_jacobian)
	q_jacobian = quote
		function $(esc(Symbol(funsym, :_jacobian)))($xsym, $psym, $(args...))
			I = Int[]
			J = Int[]
			V = Float64[]
			$body_jacobian
			return SparseArrays.sparse(I, J, V, numequations, length($xsym), +)
		end
	end
	#=
	@show MacroTools.prettify(q_residuals)
	@show MacroTools.prettify(q_jacobian)
	=#
	return quote
		$q_residuals
		$q_jacobian
	end
end

function newton(residuals, jacobian, x0; numiters=10, solver=(J, r)->J \ r)
	x = x0
	for i = 1:numiters
		J = jacobian(x)
		r = residuals(x)
		x = x - solver(J, r)
	end
	return x
end

function replace(expr, old, new)
	@MacroTools.capture(expr, $old) || return expr
	return new
end

function replaceall(expr, old, new)
	return MacroTools.postwalk(x->replace(x, old, new), expr)
end

function replacerefswithsyms(expr)
	sym2symandref = Dict()
	function replaceref(expr)
		@MacroTools.capture(expr, x_[y__]) || return expr
		sym = gensym()
		sym2symandref[sym] = (x, y)
		return sym
	end
	newexpr = MacroTools.postwalk(replaceref, expr)
	return newexpr, sym2symandref
end

function replaceaddterm(x, codegen)
	@MacroTools.capture(x, NonlinearEquations.addterm(equationnum_, term_)) || return x
	return codegen(equationnum, term)
end

function replacenumequations(x, postcode)
	@MacroTools.capture(x, NonlinearEquations.setnumequations(numequations_)) || return x
	return quote
		numequations = $numequations
		$postcode
	end
end

function replacesymswithrefs(expr, sym2symandref)
	function replacesym(expr)
		if expr isa Symbol && haskey(sym2symandref, expr)
			sym, ref = sym2symandref[expr]
			return :($sym[$(ref...)])
		else
			return expr
		end
	end
	return MacroTools.postwalk(replacesym, expr)
end

end
