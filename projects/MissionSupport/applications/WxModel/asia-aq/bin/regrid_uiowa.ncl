; extract variables from original wrfout files
; Usage: ncl 'file_in="name_in.nc"' 'file_out="name_out.nc"' regrid_uiowa.ncl
;----------------------------------------------------------------------------

; load in the libraries
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

  f=addfile(file_in, "r")
  global_attnames = getvaratts(f)
  system("rm -f "+file_out) ;-- delete file
  setfileoption("nc","Format","NetCDF4")
  fo = addfile(file_out,"c") ;-- create new

; fc = file_constants having the lat lon
fc = addfile("/discover/nobackup/mmohanty/Asia-AQ/bin/uiowa_asiaaq_fcst_constants.nc","r")
lat = fc->lat
lon = fc->lon
lev = f->lev
  minlat = min(lat)
  maxlat = max(lat)
  minlon = 88.7
  maxlon = 139.2

vars = (/"PBLH", "AOD550", "COL_OC", "COL_EC", "COL_PM25", "COL_CO", "U", "V", "T", "RH",\
	 "PM25", "EC", "OC", "SO4", "NO3", "NH4", "CO", "SO2", "O3", "CH2O",\
	"NO", "NO2"/)	

;---Set up options for regridding to 0.2 degree grid
  Opt                 = True

  Opt@SrcGridLat      = lat
  Opt@SrcGridLon      = lon

;---If you don't set these two, the regridding will be VERY slow
  Opt@DstLLCorner     = (/floor(minlat)-0.2,floor(minlon)+0.2/)
  Opt@DstURCorner     = (/ ceil(maxlat)-0.2, ceil(maxlon)+0.2/)
  Opt@DstGridType     = "0.2deg"                ; destination grid

  Opt@ForceOverwrite  = True
  Opt@Debug           = True

;---Do two types of regridding: bilinear and patch
  Opt@InterpMethod    = "neareststod"

do i = 0, dimsizes(vars)-1
    print("varname:   " + vars(i))
    v := f->$vars(i)$
    print(dimsizes(v))
        va := ESMF_regrid(v, Opt)
    fo->$vars(i)$ = va
end do
