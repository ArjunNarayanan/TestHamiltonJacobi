function plot_field(xrange,u)
    fig, ax = PyPlot.subplots()
    ax.plot(xrange,u,linewidth=3)
    ax.set_ylim([-1.5,1.5])
    ax.grid()
    return fig
end

function plot_field(xrange,numerical,axrange,analytical)
    fig,ax = PyPlot.subplots()
    ax.plot(xrange,numerical,linewidth=3,label="numerical")
    ax.plot(axrange,analytical,"--",label="analytical")
    ax.legend()
    ax.grid()
    return fig
end

function plot_field(xrange,numerical,analytical)
    return plot_field(xrange,numerical,xrange,analytical)
end

function plot_characteristic(x0;dt=1.0)
    fig, ax = PyPlot.subplots()
    xlow = minimum(x0)
    xhi = maximum(x0)
    for xi in x0
        dx = cusp_characteristic(dt,xi)
        ax.arrow(xi,0.0,dx,dt,head_width=0.02)
    end
    ax.set_xlim([xlow - 0.1*(xhi-xlow), xhi + 0.1*(xhi-xlow)])
    ax.set_ylim([0.0,1.2])
    fig.tight_layout()
    return fig
end

function mean_convergence_rate(nsplits,err)
    return mean(diff(log.(err)) ./ diff(log.(nsplits)))
end

function plot_convergence(nsplits,err)
    fig,ax = PyPlot.subplots()
    ax.loglog(nsplits,err,"-o")
    ax.grid()
    ax.set_xlabel("Number of divisions")
    ax.set_ylabel("Error vs. analytical solution")
    slope = mean_convergence_rate(nsplits,err)
    annotation = @sprintf "mean slope = %1.1f" slope
    ax.annotate(annotation, (0.2,0.2), xycoords = "axes fraction")
    return fig
end
