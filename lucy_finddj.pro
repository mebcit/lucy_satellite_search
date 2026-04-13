; be sure to have already done
;dlm_register,'/home/mbrown/idl/spice/icy/lib/icy.dlm'
; and also
; 

pro lucy_finddj,hdr,x,y,r,et

; take a headerr from LORRI and use SPICE to figure out where DJ (or something else some day)
; should be
cspice_furnsh,'/home/mbrown/lucy/gromit/sim/gromit_kernels/mk/lcy.donj.25106.v03.tm'

ti=sxpar(hdr,'midutcjd')
time=double(strmid(ti,3,30))
cspice_utc2et,string(time,format='(f25.7)')+' JD',et
cspice_spkpos,'Donaldjohanson',et,'J2000','none','Lucy',pos,ltime

r=sqrt(total(pos^2))
dec=asin(pos[2]/r)/!dtor
ra=atan2(pos[0],pos[1])/!dtor
extast,hdr,astr
ad2xy,ra,dec,astr,x,y
cspice_kclear
end
 	
