SHELL = /bin/bash
MAIN = FLEXTRA_GEOS
INCF	= incl* 
#

#INCPATH  = ${BASEDIR}/Linux/include/netcdf
#LIBPATH1 = ${BASEDIR}/Linux/lib
LIB_NETCDF := $(shell $(BASEDIR)/Linux/bin/nf-config --flibs)
INC_NETCDF := $(shell $(BASEDIR)/Linux/bin/nf-config --cflags)

#FFLAGS   =   -O2 -s -I $(INCPATH)  -m64 -mcmodel=medium
#LDFLAGS  = $(FFLAGS) -L $(LIBPATH1) -lnetcdff -lnetcdf 

FFLAGS   =   -O2 -s $(INC_NETCDF) -m64 -mcmodel=medium
LDFLAGS  = $(FFLAGS) $(LIB_NETCDF)
# for calculon
#FC := mpiifort
# for discover
FC := gfortran -fallow-argument-mismatch
#

OBJECTS = caldate.o                openoutput.o \
          checklimits.o            coordtrafo.o \
          orolininterpol.o         etatrafo.o \
          petters.o eta.ecmwf.o    FLEXTRA.o \
          pp.ecmwf.o               geteta.o \
          getfields.o              pvinterpol.o \
          getheight.o              random.o \
          readavailable.o          getwind.o \
          readcommand.o            gridcheck.geos.o \
          inter3d.o                readoro.o \
          interisentrop.o          readpaths.o \
          interisobar.o            readpoints.o \
          intermix.o               readwind.geos.o \
          intermod.o               subtractoro.o \
          interpol.o               timemanager.o \
          juldate.o                trajinterpol.o \
          lamphi_ecmwf.o           trajout.o  \
          lastprocessor.o          uncertcoor.o \
          levinterpol.o            utransform.o \
          levlininterpol.o         vtransform.o \
          lininterpol.o            wtransform.o \
          numerical.o              zztrafo.o \
          cmapf1.0.o               \
                  lininterpol_nests.o \
          levlininterpol_nests.o   orolininterpol_nests.o \
          interpol_nests.o         levinterpol_nests.o \
          skplin.o                 readcet.o \
          opencetoutput.o openflightoutput.o \
          readflight.o             calcpv.o \
          getmet.o                 calcpv_nests.o \
          swap32.o                 ew.o

# gridcheck_nests.o readwind_nests.o


$(MAIN): $(OBJECTS)
	$(FC) *.o -o $(MAIN) $(LDFLAGS)
$(OBJECTS): $(INCF)

