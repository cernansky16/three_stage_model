import numpy as np
from scipy.sparse import spdiags
import matplotlib.pyplot as plt
from itertools import product
    
    
def Qmatrix(inputlist):
    """
    Constructs a Qmatrix from a list of univariate bounded reactions.
    
    Requires:
    inputlist - a list of (bounded) univariate reactions. Each element is a pair (r,f) with
        r (int) - change in the copy number
        f - a (xmax+1)-dimensional reaction-rate vector 
    
    Ensures:
        Q - an (xmax+1)*(ymax+1) sparse transition matrix in the DIAgonal format
    """
    Q = 0.
    for r, f in inputlist:
        if r != 0:
            Qr = spdiags([-f, np.roll(f, r)], [0, r], len(f), len(f))
            Q += Qr
            
    return Q.todia()
        
def Qpacked(QT):
    """
    Creates a packed representation of a transposed transition matrix such as 
    can be fed into scipy.integrate.ode solver for sparse Jacobian
    
    Requires:
        QT - the transpose of the transition matrix in the DIAgonal format
        
    Ensures:
        M -  an (ml+mu+1, order(Q)) array, where ml and mu is the number of 
            lower and upper (nonzero) diagonals. The diagonal with offset 
            -ml <= i <= mu is stored in the row M[mu - i, :]
        ml - as described above
        mu - as described above             
    """
    ml = -min(QT.offsets)
    mu = max(QT.offsets)
    M = np.zeros(shape=(ml+mu+1, QT.shape[0]))
    M[mu - QT.offsets, :] = QT.data
    return M, ml, mu


def bnd_reaction_system(inputlist, max_values, method='stoichiometry'):
    """Bounds a Nd reaction system
    Requires:
    inputlist - a list of reactions; each element is of a form  (list_of_rates, F) with
        list_of_rates - change in species for each of the n species (integer)
        F  - reaction rate, a real-valued function of the n species, must allow for element-wise calculation
    max_values - list of maximum values (integer) for each of the n species
    method - can be 'rate' or 'stoichiometry'
    Ensures:
    A list of modified reactions. Each element is of a form (list_of_rates, F) with
        list_of_rates - change in species for each of the n species (integer)
        F - a n-dimensional array giving the reaction rates in all the possible states of cartesian product (x1max+1) x (x2max+1) ... x(xnmax+1)
    """
    grid_args = [np.arange(max_val + 1) for max_val in max_values]
    grids = np.meshgrid(*grid_args, indexing='ij')

    outputlist = []
    for reactants, rate_function in inputlist:
        F = rate_function(*grids)
        if type(F) != np.ndarray:
            F = F * np.ones(shape=np.shape(grids[0]))

        if method == 'rate':
            select = np.any([reactant + X_i > max_val for reactant, X_i, max_val in zip(reactants, grids, max_values)], axis=0)
            F[select] = 0.
            outputlist.append((reactants, F))
            
        elif method == 'stoichiometry':
            for coeffs in product(*([[r] + list(range(r)) for r in reactants])):
                if any(c != 0 for c in coeffs):
                    Ftilde = np.zeros(shape=F.shape)
                    select_conditions = [np.minimum(x_i_max - X_i, rt_x_i) == c for x_i_max, c, X_i, rt_x_i in zip(max_values, coeffs, grids, reactants)]
                    select = np.logical_and.reduce(select_conditions)
                    Ftilde[select] = F[select]
                    outputlist.append((coeffs, Ftilde))

    return outputlist
    
def flatten_multivariate_sys(inputlist):
    """Flattens a multivariate bounded reaction system into a univariate one.

    Requires:
    inputlist - a list of bounded multivariate reactions

    Ensures:
    outputlist - a list of bounded univariate reactions
    """
    ndim = len(inputlist[0][-1].shape)
    outputlist = inputlist
    for i in range(1, ndim):
        curr_input = []
        for rates, F in outputlist:
            rho = rates[0]*F.shape[i] + rates[1]
            curr_input.append(([rho, *rates[2:]], F))
        outputlist = curr_input
    
    return [(*rho, F.flatten()) for rho, F in outputlist]
   

           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
