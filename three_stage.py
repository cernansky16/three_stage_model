import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.integrate import ode
import qmatrixcreator
from binomial import *

Y_MAX = 10
Z_MAX = 1
X_INIT = 9  # protein
Y_INIT = 4  # mRNA
Z_INIT = 1  # promoter
CYCLE_DURS = [0.1, 1, 4, 10]
N_CYCLES = 4
X_MAXS = [16, 35, 70, 160]
RHO = 2
LAMBDA = 3
OFF_RATE = 0.1
ON_RATE = 0.1
DM_RATE = 1

ALL_TIMES = [
    [0.05, 0.1, 0.19, 0.2, 0.24, 0.3, 0.37],
    [0.6, 1, 1.9, 2, 2.1, 3, 3.7],
    [2.5, 4, 5, 8, 8.7, 12, 13.6],
    [6, 10, 17, 20, 23, 30, 36]
]
PLOT_TIMES = [
    [0.05, 0.19, 0.24, 0.37],
    [0.6, 1.9, 2.1, 3.7],
    [2.5, 5, 8.7, 13.6],
    [6, 17, 23, 36]
]

#Main computation
psol = {}
for cycle_index in range(N_CYCLES):
    cycle_dur = CYCLE_DURS[cycle_index]
    inits = [i * cycle_dur for i in range(N_CYCLES)]
    xmax = X_MAXS[cycle_index]
    times = ALL_TIMES[cycle_index]
    reaction_system = [
        ([0, 0, -1], lambda x, y, z: OFF_RATE * z),
        ([0, 0, 1], lambda x, y, z: ON_RATE * (1 - z)),
        ([0, 1, 0], lambda x, y, z: z * RHO),
        ([1, 0, 0], lambda x, y, z: y * LAMBDA),
        ([0, -1, 0], lambda x, y, z: DM_RATE * y)]
    
    Q = qmatrixcreator.Qmatrix(qmatrixcreator.flatten_multivariate_sys(qmatrixcreator.bnd_reaction_system(reaction_system,[xmax, Y_MAX, Z_MAX])))
    T = Q.transpose()
    Tpacked, lband, uband = qmatrixcreator.Qpacked(T)
    solver = ode(lambda t, p: T.dot(p), lambda t, p: Tpacked)
    solver.set_integrator("vode", method='bdf', with_jacobian=True, lband=lband, uband=uband)
    j = 0
    for i in range(N_CYCLES):
        if i == 0:
            probs = np.zeros((xmax + 1, Y_MAX + 1, Z_MAX + 1))
            probs[X_INIT, Y_INIT, Z_INIT] = 1
        else:
            probs = binomial_probs(probs)
        solver.set_initial_value(probs.flatten(), inits[i])
        while j < len(times):
            start = time.perf_counter()
            solver.integrate(times[j])
            end = time.perf_counter()
            # print("Ode integration took {:0.4f} seconds".format(end-start))
            psol[times[j]] = solver.y.copy()
            if times[j] == cycle_dur * (i + 1):
                probs = psol[times[j]].reshape(xmax + 1, Y_MAX + 1, Z_MAX + 1)
                j += 1
                break
            j += 1    

#Plotting results (distribution of proteins)
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(4, len(PLOT_TIMES), figsize=(2,4))
mark = [1, 2, 3, 6]
Y_LIMS = [0.6, 0.2, 0.06, 0.04]
for _ in range(4):
    for i in range(4):
        xmax = X_MAXS[_]
        ax[_, i].set_ylim(0, Y_LIMS[_])
        t = PLOT_TIMES[_][i]
        ax[_][i].plot(np.arange(xmax+1), psol[t].reshape(xmax+1, Y_MAX+1, Z_MAX+1).sum(axis=(1,2)), 
                color=cmap(i), marker='o', linestyle='-', markevery=mark[_])
        ax[_][i].legend(labels=[f"t={t:.3g}"])
        ax[_][i].legend(labels=[f"t={t:.3g}"])
    
        ax[3][i].set_xlabel("Počet proteínov")
    ax[_][0].set_ylabel("Pravdepodobnosť")
plt.show()





# """Plotting 2d distribution"""
# # Start with a square Figure.
"""THE RATE PARAMETERS AND DESIRED TIMES HAVE TO BE CHANGED IN ORDER TO OBTAIN THE SAME GRAPHS AS IN 4th chapter"""
t = ALL_TIMES[0][3] #desired time
xmax = 16
fig = plt.figure(figsize=(6, 6))
# Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.15, right=0.95, bottom=0.15, top=0.95,
                      wspace=0.15, hspace=0.15)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
p_xmarg = psol[t].reshape(xmax+1, Y_MAX+1, Z_MAX+1).sum(axis=(1,2))
ax_histx.bar(x=np.arange(len(p_xmarg)), height=p_xmarg)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
p_ymarg = psol[t].reshape(xmax+1, Y_MAX+1, Z_MAX+1).sum(axis=(0,2))
ax_histy.barh(y=np.arange(len(p_ymarg)), width=p_ymarg)
ax_histx.tick_params(axis="x", labelbottom=False)
ax_histy.tick_params(axis="y", labelleft=False)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
joint = ax.imshow(psol[t].reshape(xmax+1, Y_MAX+1, Z_MAX+1).sum(axis=2).transpose(), origin='lower')
cbbox = inset_axes(ax, '30%', '50%', loc = 1)
[cbbox.spines[k].set_visible(False) for k in cbbox.spines]
cbbox.tick_params(
    axis = 'both',
    left = False,
    top = False,
    right = False,
    bottom = False,
    labelleft = False,
    labeltop = False,
    labelright = False,
    labelbottom = False
)
cbbox.set_facecolor([1,1,1,0.7])

cbaxes = inset_axes(cbbox, '30%', '90%', loc = 6)

bar = plt.colorbar(joint, cax=cbaxes, label="Pravdepodobnosť")
print()
bar.set_ticks([round(0.1*i*np.max(psol[t].reshape(xmax+1, Y_MAX+1, Z_MAX+1)), 3) for i in range(10)])
ax.set_xlabel("Proteín $x$", size=16)
ax.set_ylabel("mRNA $y$", size=16)
ax_histx.set_ylabel("Pravdepodobnosť", size=16)
ax_histy.set_xlabel("Pravdepodobnosť", size=16)
for axi in [ax, ax_histx, ax_histy]:
    for label in axi.get_xticklabels() + axi.get_yticklabels():
        label.set_fontsize(16)
plt.show()

