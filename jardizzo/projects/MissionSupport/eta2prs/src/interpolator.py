import numpy as np

class Interpolator(object):

    def __init__(self, undef=1.0e+15):

        self.undef = undef

    def vinterp(self, var, vcoord, levels, strict=True):

        nlev = len(levels)
        km, jm, im = var.shape

        vout = np.full((nlev,jm,im), self.undef, var.dtype)
        
        for j in range(0,jm):

            for i in range(0,im):

                vc = vcoord[:,j,i].tolist()
                v  = var[:,j,i].tolist()

                for k, lev in enumerate(levels):

                    l1, l2, found = self.vfind(lev, vc)
                    if not found and strict: continue

                    vout[k,j,i] = self.interp(v[l1], v[l2], vc[l1], vc[l2], lev)

        return vout

    def vfind(self, lev, levels):

        nlev = len(levels)

        if levels[0] < levels[1]:

            if lev < levels[0]:  return (0, 1, False)
            if lev > levels[-1]: return (-2, -1, False)

            for k in range(1, nlev):
                if levels[k] >= lev: return (k-1, k, True)

        else:

            if lev > levels[0]:  return (0, 1, False)
            if lev < levels[-1]: return (-2, -1, False)

            for k in range(1, nlev):
                if levels[k] <= lev: return (k-1, k, True)

    def interp(self, q1, q2, loc1, loc2, newloc):

        slope     = (q2 - q1) / (loc2 - loc1)
        intercept = q1 - slope * loc1

        return slope * newloc + intercept
