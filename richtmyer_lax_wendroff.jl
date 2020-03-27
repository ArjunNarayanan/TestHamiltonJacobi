function step_richtmyer_lax_wendroff(u,dt,dx,flux)
    uplus = circshift(u,-1)
    umid = 0.5*(u+uplus) - 0.5*dt/dx*(flux.(uplus) - flux(u))
    umid[end] = u[end] + u[2]
    umidminus = circshift(umid,+1)
    unext = u - dt/dx*(flux.(umid)-flux(umidminus))
    return unext
end

function run_steps_lax_wendroff(sol0,dt,dx,nsteps,flux)
    sol = copy(sol0)
    for i in 1:nsteps
        sol = step_richtmyer_lax_wendroff(sol,dt,dx,flux)
    end
    return sol
end
