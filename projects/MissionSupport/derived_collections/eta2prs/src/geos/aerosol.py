import math
import numpy as np
import collections

class Aerosol(object):

    def __init__(self, fh_in, fh_alt=None, **kwargs):

        self.fh1    = fh_in
        self.fh2    = fh_alt

        self.handler = {'default'      : self.default,
                        'PM25'         : self.createPM25,
                        'PM10'         : self.createPM10,
                        'PM'           : self.createPM,
                        'DUST'         : self.createDUST,
                        'BCOC'         : self.createBCOC
                       }

    def createVariable(self, name):

        handler = self.handler.get(name, self.default)
        return handler(name)

    def default(self, name):

        Q   = self.fh1.variables[name]
        var = Q[0,:,:,:]

        attr = collections.OrderedDict(Q.__dict__)

        return (var, attr)

    def createTHETA(self, name):

        KAPPA = 0.286
        LC    = 586.0
        CP2   = 0.240

        T    = self.fh2.variables['T'][0,:,:,:]
        Q    = self.fh2.variables['QV'][0,:,:,:]
        P    = self.fh2.variables['PL'][0,:,:,:]

        theta = T * (100000.0/P)**KAPPA
        theta = theta * np.exp((LC*Q) / (CP2*T) )

        D    = self.fh2.variables['T']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'equivalent_potential_temperature',
                      'standard_name':'equivalent_potential_temperature',
                      'units':'K'}.items()) )

        return (theta, attr)

    def createPM25(self, name):

        du_coeff  = math.log(1.25)/math.log(1.8)
        no3_coeff = 80.043 / 62.0
        ss_coeff  = math.log(2.5) / math.log(3)
        so4_coeff = 132.14 / 96.06
        fn1       = 0.138

        pm25  =  self.fh1.variables['BCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['BCPHOBIC'][0,:,:,:]

        pm25 +=  self.fh1.variables['OCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['OCPHOBIC'][0,:,:,:]

        pm25 +=  self.fh1.variables['DU001'][0,:,:,:] + \
                 self.fh1.variables['DU002'][0,:,:,:] * du_coeff

        pm25 +=  self.fh1.variables['NI001'][0,:,:,:] * no3_coeff + \
                 self.fh1.variables['NI002'][0,:,:,:] * no3_coeff * fn1

        pm25 +=  self.fh1.variables['SS001'][0,:,:,:] + \
                 self.fh1.variables['SS002'][0,:,:,:] + \
                 self.fh1.variables['SS003'][0,:,:,:] * ss_coeff

        pm25 +=  self.fh1.variables['SO4'][0,:,:,:] * so4_coeff

        pm25 *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'particulate_matter_2p5_micrometers',
                      'standard_name':'particulate_matter_2p5_micrometers'
                     }.items()) )

        return (pm25, attr)

    def createPM10(self, name):

        du_coeff  = math.log(1.10)/math.log(1.8)
        du_coeff2 = math.log(1.667)/math.log(2.0)
        no3_coeff = 80.043 / 62.0
        ss_coeff  = math.log(2.5) / math.log(3)
        so4_coeff = 132.14 / 96.06
        fn1       = 0.138
        fn2       = 0.808
        fn3       = 0.164

        pm10  =  self.fh1.variables['BCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['BCPHOBIC'][0,:,:,:]

        pm10 +=  self.fh1.variables['OCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['OCPHOBIC'][0,:,:,:]

        pm10 +=  self.fh1.variables['DU001'][0,:,:,:] + \
                 self.fh1.variables['DU002'][0,:,:,:] + \
                 self.fh1.variables['DU003'][0,:,:,:] + \
                 self.fh1.variables['DU004'][0,:,:,:] * du_coeff2

        pm10 +=  self.fh1.variables['NI001'][0,:,:,:] * no3_coeff + \
                 self.fh1.variables['NI002'][0,:,:,:] * no3_coeff * fn2 + \
                 self.fh1.variables['NI003'][0,:,:,:] * no3_coeff * fn3

        pm10 +=  self.fh1.variables['SS001'][0,:,:,:] + \
                 self.fh1.variables['SS002'][0,:,:,:] + \
                 self.fh1.variables['SS003'][0,:,:,:] + \
                 self.fh1.variables['SS004'][0,:,:,:]

        pm10 +=  self.fh1.variables['SO4'][0,:,:,:] * so4_coeff

        pm10 *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'particulate_matter_10_micrometers',
                      'standard_name':'particulate_matter_10_micrometers'
                     }.items()) )

        return (pm10, attr)

    def createPM(self, name):

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

        pm +=  self.fh1.variables['NI001'][0,:,:,:] * no3_coeff + \
               self.fh1.variables['NI002'][0,:,:,:] * no3_coeff + \
               self.fh1.variables['NI003'][0,:,:,:] * no3_coeff

        pm +=  self.fh1.variables['SS001'][0,:,:,:] + \
               self.fh1.variables['SS002'][0,:,:,:] + \
               self.fh1.variables['SS003'][0,:,:,:] + \
               self.fh1.variables['SS004'][0,:,:,:] + \
               self.fh1.variables['SS005'][0,:,:,:]

        pm +=  self.fh1.variables['SO4'][0,:,:,:] * so4_coeff

        pm *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'particulate_matter_total_micrometers',
                      'standard_name':'particulate_matter_total_micrometers'
                     }.items()) )

        return (pm, attr)

    def createDUST(self, name):

        dust = self.fh1.variables['DU001'][0,:,:,:] + \
               self.fh1.variables['DU002'][0,:,:,:] + \
               self.fh1.variables['DU003'][0,:,:,:] + \
               self.fh1.variables['DU004'][0,:,:,:] + \
               self.fh1.variables['DU005'][0,:,:,:]

        dust *= self.fh1.variables['AIRDENS'][0,:,:,:]

        D    = self.fh1.variables['DU001']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'dust_total_micrometers',
                      'standard_name':'dust_total_micrometers'
                     }.items()) )

        return (dust, attr)

    def createBCOC(self, name):

        bcoc  =  self.fh1.variables['BCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['BCPHOBIC'][0,:,:,:]

        bcoc +=  self.fh1.variables['OCPHILIC'][0,:,:,:] + \
                 self.fh1.variables['OCPHOBIC'][0,:,:,:]

        bcoc *=  self.fh1.variables['AIRDENS'][0,:,:,:]

        D    = self.fh1.variables['BCPHILIC']
        attr = collections.OrderedDict(D.__dict__)
        attr.update( list({'long_name':'Black+Organic Carbon Mass',
                      'standard_name':'Black+Organic Carbon Mass',
                      'units':'kg m-3'}.items()) )

        return (bcoc, attr)
