import math
import numpy as np
import collections
import geos.interpolator as gterp

class ETA2HGT(object):

    def __init__(self, fh_in, fh_hgt, levels, strict, **kwargs):

        self.fh1    = fh_in
        self.fh2    = fh_hgt
        self.levels = list(levels)
        self.strict = strict

        self.set_const(fh_hgt.variables['lat'][:])

        if (kwargs.get('alt', False)):
            self.hgt  = self.altimetric(fh_hgt.variables['H'][0,:,:,:])
            self.phis = self.altimetric(fh_hgt.variables['PHIS'][0,:,:]/self.g0)
        else:
            self.hgt  = fh_hgt.variables['H'][0,:,:,:]
            self.phis = fh_hgt.variables['PHIS'][0,:,:] / self.g0

        if (kwargs.get('feet', False)):
            self.hgt  /= 0.3048
            self.phis /= 0.3048

        if (kwargs.get('ground', False)):
            self.hgt = self.hgt - self.phis

        self.handler = {'default'      : self.default,
                        'AGL'          : self.createAGL,
                        'THETA'        : self.createTHETA,
                        'PM25'         : self.createPM25,
                        'PM'           : self.createPM,
                        'BCOC'         : self.createBCOC,
                        'DUST'         : self.createDUST
                       }

    def set_const(self, lat):

        deg2rad  = np.pi / 180.0
        rad_lat  = lat[:] * deg2rad
        sin_lat  = np.sin(rad_lat)
        sin2_lat = np.sin(2.0*rad_lat)

        self.g0  = 9.80665

        self.zg  = 9.780356 * (1.0 + \
                               0.0052885 * sin_lat * sin_lat - \
                               0.0000059 * sin2_lat * sin2_lat)

        self.zre = 2.0 * self.zg / (3.085462e-6 + \
                                    2.27e-9 * np.cos(2.0*rad_lat) - \
                                    2.0e-12 * np.cos(4.0*rad_lat))

    def altimetric(self, geophgt):

        if len(geophgt.shape) == 2:
            return self.altimetric2D(geophgt)
        else:
            return self.altimetric3D(geophgt)

    def altimetric2D(self, geophgt):

        jm, im = geophgt.shape

        for j in range(0,jm): 
            
            g     = self.zg[j]
            re    = self.zre[j]
            ratio = g / self.g0
        
            h  = geophgt[j,:]
            geophgt[j,:] = (re * h) / (re * ratio - h)
        
        return geophgt

    def altimetric3D(self, geophgt):

        km, jm, im = geophgt.shape

        for j in range(0,jm):

            g     = self.zg[j]
            re    = self.zre[j]
            ratio = g / self.g0

            h  = geophgt[:,j,:]
            geophgt[:,j,:] = (re * h) / (re * ratio - h)

        return geophgt

    def createVariable(self, name):

        handler = self.handler.get(name, self.default)
        return handler(name)

    def default(self, name):

        transform = gterp.Interpolator()

        Q   = self.fh1.variables[name]
        q   = Q[0,:,:,:]
        var = transform.vinterp(q, self.hgt, self.levels, strict=self.strict)

        attr = collections.OrderedDict(Q.__dict__)

        return (var, attr)

    def createAGL(self, name):

        transform = gterp.Interpolator()

        agl  = self.hgt - self.phis
        var = transform.vinterp(agl, self.hgt, self.levels, strict=self.strict)

        H = self.fh2.variables['H']
        attr = collections.OrderedDict(H.__dict__)
        attr.update( list({'long_name':'above_ground_level',
                      'standard_name':'above_ground_level',
                      'units':'m'}.items()) )

        return (var, attr)

    def createTHETA(self, name):

        transform = gterp.Interpolator()

        KAPPA = 0.286
        LC    = 586.0
        CP2   = 0.240

        T    = self.fh2.variables['T'][0,:,:,:]
        Q    = self.fh2.variables['QV'][0,:,:,:]
        P    = self.fh2.variables['PL'][0,:,:,:]

        theta = T * (100000.0/P)**KAPPA
        theta = theta * np.exp((LC*Q) / (CP2*T) )
        var = transform.vinterp(theta,self.hgt, self.levels, strict=self.strict)

        D    = self.fh2.variables['T']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'equivalent_potential_temperature',
                      'standard_name':'equivalent_potential_temperature',
                      'units':'K'}.items()) )

        return (var, attr)

    def createPM25(self, name):

        transform = gterp.Interpolator()

        du_coeff  = math.log(1.25)/math.log(1.8)
        no3_coeff = 80.043 / 62.0
        ss_coeff  = math.log(2.5) / math.log(3)
        so4_coeff = 132.14 / 96.06

        pm25  =  self.fh1.variables['BCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['BCPHOBIC'][0,:,:,:]

        pm25 +=  self.fh1.variables['OCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['OCPHOBIC'][0,:,:,:]

        pm25 +=  self.fh1.variables['DU001'][0,:,:,:] + \
                 self.fh1.variables['DU002'][0,:,:,:] * du_coeff

        pm25 +=  self.fh1.variables['NO3AN1'][0,:,:,:] * no3_coeff
#                self.fh1.variables['NO3AN2'][0,:,:,:] * no3_coeff + \
#                self.fh1.variables['NO3AN3'][0,:,:,:] * no3_coeff

        pm25 +=  self.fh1.variables['SS001'][0,:,:,:] + \
                 self.fh1.variables['SS002'][0,:,:,:] + \
                 self.fh1.variables['SS003'][0,:,:,:] * ss_coeff

        pm25 +=  self.fh1.variables['SO4'][0,:,:,:] * so4_coeff

        pm25 *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        var = transform.vinterp(pm25, self.hgt, self.levels, strict=self.strict)

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'particulate_matter_2p5_micrometers',
                      'standard_name':'particulate_matter_2p5_micrometers'
                     }.items()) )

        return (var, attr)

    def createPM(self, name):

        transform = gterp.Interpolator()

        du_coeff  = math.log(1.25)/math.log(1.8)
        no3_coeff = 80.043 / 62.0
        ss_coeff  = math.log(2.5) / math.log(3)
        so4_coeff = 132.14 / 96.06

        pm  =  self.fh1.variables['BCPHILIC'][0,:,:,:] + \
               self.fh1.variables['BCPHOBIC'][0,:,:,:]

        pm +=  self.fh1.variables['OCPHILIC'][0,:,:,:] + \
               self.fh1.variables['OCPHOBIC'][0,:,:,:]

        pm +=  self.fh1.variables['DU001'][0,:,:,:] + \
               self.fh1.variables['DU002'][0,:,:,:] + \
               self.fh1.variables['DU003'][0,:,:,:] + \
               self.fh1.variables['DU004'][0,:,:,:] + \
               self.fh1.variables['DU005'][0,:,:,:]

        pm +=  self.fh1.variables['NO3AN1'][0,:,:,:] * no3_coeff + \
               self.fh1.variables['NO3AN2'][0,:,:,:] * no3_coeff + \
               self.fh1.variables['NO3AN3'][0,:,:,:] * no3_coeff

        pm +=  self.fh1.variables['SS001'][0,:,:,:] + \
               self.fh1.variables['SS002'][0,:,:,:] + \
               self.fh1.variables['SS003'][0,:,:,:] + \
               self.fh1.variables['SS004'][0,:,:,:] + \
               self.fh1.variables['SS005'][0,:,:,:]

        pm +=  self.fh1.variables['SO4'][0,:,:,:] * so4_coeff

        pm *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        var = transform.vinterp(pm, self.hgt, self.levels, strict=self.strict)

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'particulate_matter_total_micrometers',
                      'standard_name':'particulate_matter_total_micrometers'
                     }.items()) )

        return (var, attr)

    def createDUST(self, name):

        transform = gterp.Interpolator()

        dust = self.fh1.variables['DU001'][0,:,:,:] + \
               self.fh1.variables['DU002'][0,:,:,:] + \
               self.fh1.variables['DU003'][0,:,:,:] + \
               self.fh1.variables['DU004'][0,:,:,:] + \
               self.fh1.variables['DU005'][0,:,:,:]

        dust *= self.fh1.variables['AIRDENS'][0,:,:,:]

        var = transform.vinterp(dust, self.hgt, self.levels, strict=self.strict)

        D    = self.fh1.variables['DU001']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'dust_total_micrometers',
                      'standard_name':'dust_total_micrometers'
                     }.items()) )

        return (var, attr)

    def createBCOC(self, name):

        transform = gterp.Interpolator()

        bcoc  =  self.fh1.variables['BCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['BCPHOBIC'][0,:,:,:]

        bcoc +=  self.fh1.variables['OCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['OCPHOBIC'][0,:,:,:]

        bcoc *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        var = transform.vinterp(bcoc, self.hgt, self.levels, strict=self.strict)

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'Black+Organic Carbon Mass',
                      'standard_name':'Black+Organic Carbon Mass',
                      'units':'kg m-3'}.items()) )

        return (var, attr)
