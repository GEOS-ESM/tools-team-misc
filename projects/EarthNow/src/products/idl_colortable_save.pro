PRO idl_colortable_save, ctable, REVERSE=reverse

  ; Load the requested IDL color table
  LOADCT, ctable

  ; Get RGB vectors
  TVLCT, r, g, b, /GET

  ; Build output filename
  suffix = '_normal'
  IF KEYWORD_SET(reverse) THEN suffix = '_reversed'

  filename = '/discover/nobackup/projects/gmao/g6dev/pub/ColorTables/idl_colortable_' + STRTRIM(ctable, 2) + suffix + '.txt'

  ; Open file for writing
  OPENW, lun, filename, /GET_LUN

  ; Write colors
  IF KEYWORD_SET(reverse) THEN BEGIN
    FOR i = N_ELEMENTS(r)-1, 0, -1 DO $
      PRINTF, lun, r[i], g[i], b[i]
  ENDIF ELSE BEGIN
    FOR i = 0, N_ELEMENTS(r)-1 DO $
      PRINTF, lun, r[i], g[i], b[i]
  ENDELSE

  ; Close file
  FREE_LUN, lun

END

