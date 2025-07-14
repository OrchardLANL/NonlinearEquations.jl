module NonlinearEquations

include("Calculus.jl")
import MacroTools
import SparseArrays
import ChainRulesCore

function codegen_addterm_residuals(equationnum, term)
	return quote 
		residuals[$equationnum] += $term
	end
end

function codegen_addterm_print_equation(equationnum, term)
	return quote 
        if ___equation_number___ == $equationnum
            if ___is_first_term___
                ___equation_string_values___ = string("{", $term, "}")
                ___equation_string_symbols___ = string("{", $(string(term)), "}")
                ___is_first_term___ = false
            else
                ___equation_string_values___ *= string(" + {", $term, "}")
                ___equation_string_symbols___ *= string(" + {", $(string(term)), "}")
            end
        end
	end
end

function codegen_addterm_inplacejacobian(equationnum, term, xsym)
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
				___jacobian_storage___[$equationnum, $ref] += $derivative
			end
			append!(q.args, newcode.args)
		end
		return q
	else
		return :()
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
	if typeof(ex) == Symbol#this hack gets around the fact that Calculus.differentiate doesn't have a function differentiate(::Symbol, ::Array{Symbol,1})
		diffs = Any[]
		for i = 1:length(diffsyms)
			if diffsyms[i] == ex
				push!(diffs, :(1))
			else
				push!(diffs, :(0))
			end
		end
	else
		diffs = Calculus.differentiate(ex, diffsyms)
	end
	diffs = map(diff->Calculus.simplify(replacesymswithrefs(diff, dict)), diffs)
	return diffs, diffrefs
end

macro equations(other, fundef)
	if @MacroTools.capture(other, expandmodule=modulename_)
		return equations(fundef, eval(modulename), Symbol[])
	elseif @MacroTools.capture(other, exclude=(varnames_))
		return equations(fundef, Main, map(x->x, varnames.args))
	end
	return equations(fundef, Main, Symbol[])
end

macro equations(fundef)
	return equations(fundef, Main, Symbol[])
end

function equations(fundef::Expr, macroexpand_module, dont_differentiate_syms::Array{Symbol, 1})
        dict = MacroTools.splitdef(fundef)
        original_args_list = copy(dict[:args])
        original_kwargs_list = copy(dict[:kwargs])
        original_whereparams = get(dict, :whereparams, nothing)
        original_arg_names = map(arg->MacroTools.splitarg(arg)[1], original_args_list)
        diff_arg_names = [MacroTools.splitarg(a)[1] for a in filter(a->!(MacroTools.splitarg(a)[1] in dont_differentiate_syms), original_args_list)]
        original_body = macroexpand(macroexpand_module, dict[:body])
	#generate the code for computing the residuals
	body_residuals = MacroTools.postwalk(x->replacenumequations(x, :(residuals = zeros(numequations))), original_body)
	body_residuals = MacroTools.postwalk(x->replaceaddterm(x, codegen_addterm_residuals), body_residuals)
	original_name = dict[:name]
	dict[:name] = Symbol(original_name, :_residuals)
	dict[:body] = quote
		$body_residuals
		return residuals
	end
	q_residuals = MacroTools.combinedef(dict)
	q_result = quote
		$(esc(q_residuals))
	end
	#generate the code for printing equations
	pushfirst!(dict[:args], :___equation_number___)
    body_print_equation = MacroTools.postwalk(x->replacenumequations(x, :()), original_body)
    body_print_equation = MacroTools.postwalk(x->replaceaddterm(x, (eqnum, term)->codegen_addterm_print_equation(eqnum, term)), body_print_equation)
    dict[:name] = Symbol(original_name, :_print_equation)
    dict[:body] = quote
        ___equation_string_values___ = ""
        ___equation_string_symbols___ = ""
        ___is_first_term___ = true
        $body_print_equation
        println(___equation_string_symbols___)
        println(___equation_string_values___)
        println("printed equatoin string")
        return nothing
    end
    push!(q_result.args, :($(esc(MacroTools.combinedef(dict)))))
    dict[:args] = dict[:args][2:end]
	#generate the code for the jacobian
	for arg in filter(x->!(x in dont_differentiate_syms), dict[:args])
		arg_name = MacroTools.splitarg(arg)[1]
		body_jacobian = MacroTools.postwalk(x->replacenumequations(x, :()), original_body)
		body_jacobian = MacroTools.postwalk(x->replaceaddterm(x, (eqnum, term)->codegen_addterm_jacobian(eqnum, term, arg_name)), body_jacobian)
		dict[:name] = Symbol(original_name, :_, arg_name)
		dict[:body] = quote
			I = Int[]
			J = Int[]
			V = eltype($arg)[]
			$body_jacobian
			return SparseArrays.sparse(I, J, V, numequations, length($arg_name), +)
		end
		push!(q_result.args, :($(esc(MacroTools.combinedef(dict)))))
	end
	#generate the in-place versions of the jacobian
	pushfirst!(dict[:args], :___jacobian_storage___)
        for arg in filter(x->!(x in dont_differentiate_syms), dict[:args])
                arg_name = MacroTools.splitarg(arg)[1]
                body_inplacejacobian = MacroTools.postwalk(x->replacenumequations(x, :()), original_body)
                body_inplacejacobian = MacroTools.postwalk(x->replaceaddterm(x, (eqnum, term)->codegen_addterm_inplacejacobian(eqnum, term, arg_name)), body_inplacejacobian)
                dict[:name] = Symbol(original_name, :_, arg_name, :!)
                dict[:body] = quote
                        fill!(SparseArrays.nonzeros(___jacobian_storage___), zero(eltype($arg)))
                        $body_inplacejacobian
                        return nothing
                end
                push!(q_result.args, :($(esc(MacroTools.combinedef(dict)))))
        end

        #generate the ChainRulesCore rrule for the residuals
        dict[:args] = [:(::typeof($(Symbol(original_name, :_residuals)))), original_args_list...]
        dict[:kwargs] = original_kwargs_list
        dict[:whereparams] = original_whereparams
        dict[:name] = :(ChainRulesCore.rrule)

        call_residuals = Expr(:call, Symbol(original_name, :_residuals), original_arg_names..., original_kwargs_list)
        pullback_ex = Expr(:block)
        for name in original_arg_names
                if name in diff_arg_names
                        jac_call = Expr(:call, Symbol(original_name, :_, name), original_arg_names..., original_kwargs_list)
                        push!(pullback_ex.args, :(d_$(name) = transpose($jac_call) * ȳ))
                else
                        push!(pullback_ex.args, :(d_$(name) = ChainRulesCore.NoTangent()))
                end
        end
        retvals = [:ChainRulesCore.NoTangent()]
        for name in original_arg_names
                push!(retvals, Symbol(:d_, name))
        end
        push!(retvals, :(ChainRulesCore.NoTangent()))
        push!(pullback_ex.args, Expr(:return, Expr(:tuple, retvals...)))

        dict[:body] = quote
                y = $call_residuals
                function pullback(ȳ)
                        $pullback_ex
                end
                return y, pullback
        end
        push!(q_result.args, :($(esc(MacroTools.combinedef(dict)))))
        #display(MacroTools.prettify(q_result))
        return q_result
end

function escapesymbols(expr, symbols)
	for symbol in symbols
		expr = NonlinearEquations.replaceall(expr, symbol, Expr(:escape, symbol))
	end
	return expr
end

function gradient(x, p, g_x, g_p, f_x, f_p)
	lambda = transpose(f_x(x, p)) \ g_x(x, p)
	return g_p(x, p) - transpose(f_p(x, p)) * lambda
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
