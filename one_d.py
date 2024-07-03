import numpy as np
import matplotlib.pyplot as plt
import qmatrixcreator
from scipy.sparse import linalg

XMAXS = [9, 14, 20, 60, 100]
OMEGAS = [3, 5, 10, 30, 50]

def normalise(p):
    return p/p.sum()

fig, ax = plt.subplots(1, 1, figsize=(4, 3))
for i in range(len(OMEGAS)):
    omega = OMEGAS[i]
    xmax = XMAXS[i]
    rsystem = [([1], lambda x : omega),
            ([-2], lambda x: (x*(x-1))/(2*omega))]

    Q = qmatrixcreator.Qmatrix(qmatrixcreator.flatten_multivariate_sys(qmatrixcreator.bnd_reaction_system(rsystem,[xmax])))

    pnum = normalise(linalg.eigs(Q.transpose(), k=1, which='LR')[1])
    cmap = plt.get_cmap("tab10")
    numline, = ax.plot(np.arange(xmax+1)/omega, pnum, color=cmap(i), ls='-', marker='o', ms=3, alpha=0.7)
    ax.set_xlabel("$X$/$\Omega$")
    ax.set_ylabel("Pravdepodobnosť")

ax.legend(labels=["$\Omega$ = "+str(omega) for omega in OMEGAS], fontsize='medium')
        
plt.tight_layout()
plt.show()


"""Plotting Q matrix"""
Qarray = Q.toarray()  
fig, ax = plt.subplots(1,1, figsize=(4,3))
axislabel = "Entita $X$"
ax.set_xlabel(axislabel)
ax.set_ylabel(axislabel)
ax.spy(Q > 0, ms=5, c='k')
ax.spy(Q < 0, ms=5, c='r')
for pos in np.arange(xmax+1):
    ax.axhline(pos, ls=':', c='0.5')
    ax.axvline(pos, ls=':', c='0.5')
plt.tight_layout()
plt.show() 

