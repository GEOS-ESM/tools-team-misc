set called = ( $_ )
if ( "$called" != "" ) then
    set script_path = $called[2]
    set script_dir = `dirname $script_path`
endif

source /usr/share/lmod/lmod/init/csh
module load python/GEOSpyD
module load ffmpeg

set INSTALL_PATH = `realpath $script_dir/../`
setenv PYTHONPATH "$INSTALL_PATH/src:$INSTALL_PATH/cylc/src"
setenv PATH "$INSTALL_PATH/bin:$INSTALL_PATH/cylc/bin:$PATH"

umask 022
