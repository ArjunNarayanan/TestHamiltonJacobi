function step_lax_friedrichs(u,j,N,dt_by_dx,flux)
    uplus = 0.0
    uminus = 0.0
    if j == 1 || j == N
        uplus = u[2]
        uminus = u[N-1]
    else
        uplus = u[j+1]
        uminus = u[j-1]
    end
    return 0.5*(uplus+uminus - dt_by_dx*(flux(uplus) - flux(uminus)))
end

function step_lax_friedrichs(u,dt_by_dx,flux)
    ndofs = length(u)
    unext = zeros(ndofs)
    for j in 1:ndofs
        unext[j] = step_lax_friedrichs(u,j,ndofs,dt_by_dx,flux)
    end
    return unext
end

function run_steps_lax_friedrichs(sol0,dt,dx,nsteps,flux)
    sol = copy(sol0)
    dt_by_dx = dt/dx
    for i in 1:nsteps
        sol = step_lax_friedrichs(sol,dt_by_dx,flux)
    end
    return sol
end
