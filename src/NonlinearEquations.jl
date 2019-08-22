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
	if MacroTools.inexpr(term, xsym)
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
	else
		return :()
	end
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
	dict = MacroTools.splitdef(fundef)
	original_body = macroexpand(Main, dict[:body])
	body_residuals = MacroTools.postwalk(x->replacenumequations(x, :(residuals = zeros(numequations))), original_body)
	body_residuals = MacroTools.postwalk(x->replaceaddterm(x, codegen_addterm_residuals), body_residuals)
	original_name = dict[:name]
	dict[:name] = Symbol(original_name, :_residuals)
	dict[:body] = quote
		$body_residuals
		return residuals
	end
	q_residuals = MacroTools.combinedef(dict)
	body_jacobian = MacroTools.postwalk(x->replacenumequations(x, :()), original_body)
	body_jacobian = MacroTools.postwalk(x->replaceaddterm(x, (eqnum, term)->codegen_addterm_jacobian(eqnum, term, MacroTools.splitarg(dict[:args][1])[1])), body_jacobian)
	dict[:name] = Symbol(original_name, :_jacobian)
	dict[:body] = quote
		I = Int[]
		J = Int[]
		V = Float64[]
		$body_jacobian
		return SparseArrays.sparse(I, J, V, numequations, length($(MacroTools.splitarg(dict[:args][1])[1])), +)
	end
	q_jacobian = MacroTools.combinedef(dict)
	#@show MacroTools.prettify(q_residuals)
	#@show MacroTools.prettify(q_jacobian)
	return quote
		$(esc(q_residuals))
		$(esc(q_jacobian))
	end
end

function escapesymbols(expr, symbols)
	for symbol in symbols
		expr = NonlinearEquations.replaceall(expr, symbol, Expr(:escape, symbol))
	end
	return expr
end

function newtonish(residuals, jacobian, x0; numiters=10, solver=(J, r)->J \ r, rate=0.05, callback=(x, r, J, i)->nothing)
	x = x0
	for i = 1:numiters
		J = jacobian(x)
		r = residuals(x)
		#x = (1 - rate) * x - rate * solver(J, r)
		x = x - rate * solver(J, r)
		callback(x, r, J, i)
	end
	return x
end

function newton(residuals, jacobian, x0; numiters=10, solver=(J, r)->J \ r, callback=(x, r, J, i)->nothing)
	x = x0
	for i = 1:numiters
		J = jacobian(x)
		r = residuals(x)
		x = x - solver(J, r)
		callback(x, r, J, i)
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
	newexpr = MacroTools.prewalk(replaceref, expr)
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

function updateentries!(dest::SparseArrays.SparseMatrixCSC, src::SparseArrays.SparseMatrixCSC)
	if dest.colptr != src.colptr || dest.rowval != src.rowval
		error("Cannot update entries unless the two matrices have the same pattern of nonzeros")
	end
	copy!(dest.nzval, src.nzval)
end

end
