import NonlinearEquations
import PyPlot
import SparseArrays

#this block corresponds to the "Parameter values" table
D = 1e-9#m^2/s
M = 4.40098e-2#kg/mol
q = 1.5491e-10#m/s
V_ϕ = 34.8395e-6#m^3/mol
κ = 1e-13#m^2
μ = 6.7549e-5#kg/m/s
ρ_s = 1072.4#kg/m^3
ρ_w = 1066.5#kg/m^3
θ = 0.0844#unitless
g = 9.8067#m/s
Λ = κ / (μ * θ)
Γ = -g * (M - ρ_w * V_ϕ)

#this block corresponds to the "Initial conditions" table
#nx = 1000
#nz = 100
#nx = 50
#nz = 5
#nx = 100
#nz = 10
nx = 100
nz = 10
mins = [0, 0]
maxs = [1000, 100]
deltax = (maxs[1] - mins[1]) / nx
deltaz = (maxs[2] - mins[2]) / nz
tfinal = 3.1536e10#seconds
c0 = [j == nz && i * deltax >= 100 && i * deltax <= 500 ? 750.6 : 4.4457e-2 for j = 1:nz, i = 1:nx]#this is similar to the initial conditions in the document, but matches the dirichlet boundary condition

function plot(args...; filename=false)
	numplots = div(length(args), 2)
	fig, axs = PyPlot.subplots(1, numplots, figsize=(4 * numplots + 1, 4))
	for i = 1:numplots
		x = args[(i - 1) * 2 + 1]
		title = args[(i - 1) * 2 + 2]
		if startswith(title, "log")
			img = axs[i].imshow(x, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], 10 * maxs[2]], vmin=-3)
		else
			img = axs[i].imshow(x, origin="lower", interpolation="nearest", extent=[mins[1], maxs[1], mins[2], 10 * maxs[2]])
		end
		axs[i].set(xlabel="x [meters]", ylabel="z [decimeters]")
		axs[i].set_title(title)
		fig.colorbar(img, ax=axs[i])
	end
	fig.tight_layout()
	display(fig)
	println()
	if  filename != false
		fig.savefig(filename)
	end
	PyPlot.close(fig)
end

#maybe the key is just that it is diffusing in for their example. change the transport boundary conditions so that there’s no CO2 in the reservoir initially, and it only diffuses in
#still, something weird is going on with the pressure -- gravity seems to be absent from the pressure

z(j) = j * deltaz
ij2k(i, j) = j + (i - 1) * nz
@NonlinearEquations.equations function pressure(p, c)
	NonlinearEquations.setnumequations(length(p))
	#do the internal nodes
	for i = 2:nx - 1, j = 1:nz
		k = ij2k(i, j)
		ju = j < nz ? j + 1 : j
		jd = j > 1 ? j - 1 : j
		il = i > 1 ? i - 1 : i
		ir = i < nx ? i + 1 : i
		kl = ij2k(il, j)
		kr = ij2k(ir, j)
		kd = ij2k(i, jd)
		ku = ij2k(i, ju)
		NonlinearEquations.addterm(k, (p[kr] - 2 * p[k] + p[kl]) / deltax ^ 2)
		NonlinearEquations.addterm(k, (p[ku] - 2 * p[k] + p[kd]) / deltaz ^ 2)
		NonlinearEquations.addterm(k, ((ρ_w * g - Γ * c[ku])* (ju - j) - (ρ_w * g - Γ * c[k]) * (j - jd)) * deltaz / deltaz ^ 2)#gravity term
	end
	#do the bottom nodes on the east/west boundary
	k_east = ij2k(nx, nz)
	k_west = ij2k(1, nz)
	#NonlinearEquations.addterm(k_east, p[k_east] - 1.99998975026e7)
	NonlinearEquations.addterm(k_east, p[k_east] - 2e7)
	NonlinearEquations.addterm(k_west, p[k_west] - 2e7)
	#do the east/west boundaries
	for j = 1:nz - 1
		k_east = ij2k(nx, j)
		ku_east = ij2k(nx, j + 1)
		k_west = ij2k(1, j)
		ku_west = ij2k(1, j + 1)
		#NonlinearEquations.addterm(k_east, p[k_east] - p[ku_east] - g * (ρ_w * (1 - c[k_east] * V_ϕ) + M * c[k_east]) * deltaz)
		#NonlinearEquations.addterm(k_west, p[k_west] - p[ku_west] - g * (ρ_w * (1 - c[k_west] * V_ϕ) + M * c[k_west]) * deltaz)
		#NonlinearEquations.addterm(k_west, p[k_west] - g * ρ_w * (nz - 1 + j) * deltaz)
		#NonlinearEquations.addterm(k_east, p[k_east] + g * ρ_w * (nz - 1 + j) * deltaz)
		NonlinearEquations.addterm(k_west, p[k_west] - 2e7 - g * ρ_w * (nz - 1 - j) * deltaz)
		NonlinearEquations.addterm(k_east, p[k_east] - 2e7 - g * ρ_w * (nz - 1 - j) * deltaz)
	end
end
@NonlinearEquations.equations function concentration(p, c)
	NonlinearEquations.setnumequations(length(c))
	cell_volume = deltax * deltaz * 1#assume it is 1m in the y direction
	areax = deltaz * 1
	areaz = deltax * 1
	for i = 1:nx, j = 1:nz
		k = ij2k(i, j)
		if j == nz && i * deltax >= 100 && i * deltax <= 500
			NonlinearEquations.addterm(k, c[k] - 750.6)
		else
			kl = i > 1 ? ij2k(i - 1, j) : k#node to the left
			kr = i < nx ? ij2k(i + 1, j) : k#node to the right
			kd = j > 1 ? ij2k(i, j - 1) : k#node below
			ku = j < nz ? ij2k(i, j + 1) : k#node above
			for ko in [kl, kr]
				k_upstream = p[ko] - p[k] > 0 ? ko : k
				NonlinearEquations.addterm(k, -Λ * (p[ko] - p[k]) / deltax * c[k_upstream] * areax / cell_volume)
			end
			for (ko, gravity_sign) in zip([kd, ku], [1, -1])
				k_viscous_upstream = p[ko] - p[k] > 0 ? ko : k
				NonlinearEquations.addterm(k, -Λ * (p[ko] - p[k]) / deltaz * c[k_viscous_upstream] * areaz / cell_volume)
				if k != ko#don't let gravity move stuff across the boundary
					k_gravity_upstream = gravity_sign == 1 ? k : ko
					NonlinearEquations.addterm(k, gravity_sign * Λ * (g * ρ_w - Γ * c[k_gravity_upstream]) / deltaz * c[k_gravity_upstream] * areaz / cell_volume)
				end
			end
		end
	end
end

#do explicit time stepping
dt = 1e4 / nz^2 * 10^2
c = c0# .+ 1e-3 * rand(size(c0))
p = zeros(size(c0))
t = 0.0
i = 0
frame_counter = 1
#=
	c = zeros(nz, nx)
	A = pressure_p(zeros(nz, nx), c)
	b = -pressure_residuals(zeros(nz, nx), c)
	p = reshape(A \ b, nz, nx)
	@show hash(p)
	plot(log10.(c), "log10(concentration)", p, "pressure")
	error("blah")
	=#
A = pressure_p(zeros(nz, nx), c0)
b = -pressure_residuals(zeros(nz, nx), c0)
p0 = reshape(A \ b, nz, nx)
p = p0
while t < 365 * 10
	global c, t, p, i, frame_counter
	A = pressure_p(zeros(nz, nx), c)
	b = -pressure_residuals(zeros(nz, nx), c)
	lastp = p
	p = reshape(A \ b, nz, nx)
	dc = reshape(-concentration_residuals(p, c), nz, nx)
	lastc = c
	c += dt * dc
	t += dt / 3600 / 24
	if mod(i, 100) == 0
		tstring = round(t; digits=1)
		plot(c, "log10(concentration) at t=$tstring", p, "pressure", p - p0, "p-p0", c - c0, "c-c0", p - lastp, "dp", c - lastc, "dc")
		frame_counter += 1
	end
	i += 1
end
tstring = round(t; digits=1)
plot(log10.(c), "log10(concentration) at t=$tstring days", p, "pressure at t=$tstring days", c - c0, "c-c0")
#plot(log10.(c), "log10(concentration) at t=$tstring days", p, "pressure at t=$tstring days"; filename="figs/frame_$(lpad(frame_counter, 6, '0')).png")
#run(`ffmpeg -y -r 25 -f image2 -s 1920x1080 -i figs/frame_%06d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p movie.mp4`)
