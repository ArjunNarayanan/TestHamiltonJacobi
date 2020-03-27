using LinearAlgebra, PyPlot, Statistics, Printf
include("lax_friedrichs.jl")
include("richtmyer_lax_wendroff.jl")
include("plot_utils.jl")

function cusp_initial_condition(x)
    return sin(2*pi*x)
end

function cusp_flux(p)
    return -V*sqrt(1.0+p^2)/2pi
end

function cusp_characteristic_eq(xi,t,x)
    return x - xi + V*t*sin(xi)/sqrt(1+sin(xi)^2)
end

function cusp_characteristic_curve(xi,t)
    return xi - V*t*sin(xi)/sqrt(1+sin(xi)^2)
end

function cusp_characteristic_derivative(xi,t)
    return -1.0 + V*t*cos(xi)/(1+sin(xi)^2)^1.5
end

function newton_iterate(F,Fprime,x0;tol=1e-8,atol=1e-10)
    xnext = x0
    normalizer = abs(F(xnext))
    if normalizer < atol
        return xnext
    else
        val = F(xnext)
        dval = Fprime(xnext)
        err = abs(val)/normalizer
        while err > tol
            xnext = xnext - val/dval
            val = F(xnext)
            dval = Fprime(xnext)
            err = abs(val)/normalizer
        end
        return xnext
    end
end

function cusp_analytical_solution(xrange,t)
    sol = zeros(length(xrange))
    guess = xrange[1]
    for (idx,x) in enumerate(xrange)
        F(xi) = cusp_characteristic_eq(xi,t,2pi*x)
        Fprime(xi) = cusp_characteristic_derivative(xi,t)
        xi = newton_iterate(F,Fprime,guess)
        sol[idx] = sin(xi)
        guess = xi
    end
    return sol
end

function cusp_maximum_velocity()
    return abs(V/(2pi*sqrt(2)))
end

function bisect(constraint,start)
    dt = start
    nsteps = 1
    while dt > constraint
        dt = 0.5*dt
        nsteps = 2*nsteps
    end
    return dt,nsteps
end

function step_size_and_nsteps(dx)
    Vmax = cusp_maximum_velocity()
    @assert Vmax ≈ 1.0
    dt = CFL*dx
    nsteps = final_time/dt
    nsteps_int = round(Int,nsteps)
    if abs(nsteps - nsteps_int) > 1e-2
        error("Time step size $dt is inconsistent with final_time $final_time")
    end
    return dt,nsteps_int
end

function numerical_solution(nsplits::Int)
    ndofs = nsplits+1
    dx = (xR - xL)/nsplits
    dt,nsteps = step_size_and_nsteps(dx)
    xrange = range(xL,stop=xR,length=ndofs)
    sol0 = cusp_initial_condition.(xrange)
    sol = run_steps_lax_friedrichs(sol0,dt,dx,nsteps,cusp_flux)
    return sol,xrange
end

function error_for_splits(nsplits::Int)
    sol,xrange = numerical_solution(nsplits)
    analytical_solution = cusp_analytical_solution(xrange,final_time)
    return norm(sol-analytical_solution)/length(sol)
end

const xL = -0.5
const xR = 0.5
const CFL = 1.0
const V = 2pi*sqrt(2)
const final_time = 0.15

sol,xrange = numerical_solution(200)
asol = cusp_analytical_solution(xrange,final_time)
plot_field(2pi*xrange,sol,asol)
# nsplits = [100,200,500,1000,2000,5000]
# err = error_for_splits.(nsplits)
# plot_convergence(nsplits,err)

# nsplits = 10
# ndofs = nsplits+1
# dx = (xR - xL)/nsplits
# dt,nsteps = step_size_and_nsteps(dx)
# xrange = range(xL,stop=xR,length=ndofs)
# sol0 = cusp_initial_condition.(xrange)
# sol = run_steps_lax_friedrichs(sol0,dt,dx,nsteps,cusp_flux)
