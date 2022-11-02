#!/bin/csh
# run this script to set up a xtb environment
# requirements: $XTBHOME is set to `pwd`
if ( ! ${?XTBHOME} ) then
   setenv XTBHOME "`dirname $0`/../../"
endif

# set up path for xtb, using the xtb directory and the users home directory
setenv XTBPATH ${XTBHOME}/share/xtb:${HOME}

# to include the documentation we include our man pages in the users manpath
setenv MANPATH ${MANPATH}:${XTBHOME}/share/man

# finally we have to make the binaries and scripts accessable
setenv PATH ${PATH}:${XTBHOME}/bin

# enable package config for xtb
setenv PKG_CONFIG_PATH ${PKG_CONFIG_PATH}:${XTBHOME}/lib/pkgconfig
