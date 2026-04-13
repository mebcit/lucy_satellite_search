; be sure to have already done
;dlm_register,'/home/mbrown/idl/spice/icy/lib/icy.dlm'
; and also
; 

pro getgeometry,hdr,range,phase,delta,ra,dec,target=target
; range and (heliocentric) delta are in km
; phase in degrees

if n_params() eq 0 then begin
	print,'pro getgeometry,hdr,range,phase,delta,ra,dec,target=target'
	retall
endif


; take a headerr from LORRI and use SPICE to figure out 
; the range and phase angle to the target

if keyword_set(target) eq 0 then target='DonaldJohanson'

if target eq 'DonaldJohanson' then begin
	cspice_furnsh,'/home/mbrown/lucy/kernels/mk/lcy.donj.science.LATEST.tm'
	target='DonaldJohanson'
endif else begin
	message,'no spice kernel available for '+target
endelse


ti=sxpar(hdr,'midutcjd')
time=double(strmid(ti,3,30))
cspice_utc2et,string(time,format='(f25.7)')+' JD',et
cspice_spkpos,target,et,'J2000','none','Lucy',pos,ltime
cspice_spkpos,target,et,'J2000','none','Sun',spos,ltime
range=sqrt(total(pos^2))
delta=sqrt(total(spos^2))
phase=acos(total(pos*spos)/range/delta)/!dtor
dec=asin(pos[2]/range)/!dtor
ra=atan2(pos[0],pos[1])/!dtor
cspice_kclear
end
 	
