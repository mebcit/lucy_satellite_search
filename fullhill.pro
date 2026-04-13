dlm_register,'/home/mbrown/idl/spice/icy/lib/icy.dlm'
closest=798443545.D0 ; from the SPICE kernel


diametersat=20.
albedosat=0.041
satdist=200.
satang=randomu(iseed)*360

; use sat dist to figure out a circular orbital velocity
; and apply it to satang. omega=sqrt(GM/R)

volume=58.3 ; km^3
mass=volume*1.2/1000.*1.e15
hillsphere= 2.064*1.5e8*(mass/2.e30/3.)^.333

omega=sqrt(6.64e-11*mass/satdist/1000.)/satdist/1000.


;omega is in radians per second
; imb is the real data
; imbs scales to a constant DJ brightness (but not distance)
; imz scale to constant scale and brightness

; full hill sphere!


;currently assuming 8.8x3.5(x3.5) km, density=1.2
; volume is thus 30.7 km^3 or 30.7*(100000.)^3 cm^3i
; so the mass is 3.7e16 g = 3.7e13 kg


;dj is 2.064 AU away

djx=479 ; manual determined position of image zero
djy=551


fs=file_search('../llori/2025110/lor*fit',count=count)
files=[4418,4422,4426,4430,4434,4438,4442,4446,4450,4454]
num=fix(strmid(file_basename(fs),15,5))
readcol,'fullhillshifts.dat',ii,xs,ys

imb=fltarr(1024,1024,10)
ims=imb
imsat=imb
imsats=imb

rr=fltarr(10)
dj=fltarr(10)
sec=fltarr(10)
kpp=sec
dd=sec
pp=sec

imbs=imb
readcol,'fullhillshifts.dat',ii,xs,ys
for i=0,9 do begin
	w=where(files[i] eq num)
	imb[*,*,i]=xyshift(readfits(fs[w[0]],h),xs[i],ys[i])
	lucy_getpsf,imb[*,*,i],psf=psf
	sky,imb[*,*,i],m
	imb[*,*,i]-=m
	ims[*,*,i]=readfits(fs[w[0]-2],hs)*100. ; get the position and photometry
	if i eq 0 then begin
		im0=ims[*,*,0]
	endif else begin
		findshift,ims[*,*,i],im0,x,y
		ims[*,*,i]=xyshift(ims[*,*,i],-x,-y)
	endelse

	sky,ims[*,*,i],m
	ims[*,*,i]-=m

	getgeometry,h,range,phase,delta
	kpp[i]=.93/3600.*!dtor*range
	rr[i]=range
	dd[i]=delta
	pp[i]=phase
	sec[i]=sxpar(hs,'midsclk')
	dj[i]=total(ims[430:538,479:618,i])
	print,(sec[i]-closest)/60.,range

	satx=satdist*cos(satang*!dtor+(sec[i]-sec[0])*omega)
	saty=satdist*sin(satang*!dtor+(sec[i]-sec[0])*omega)

	flux=fakesat(diametersat,albedosat,range,delta,phase)*sxpar(h,'exptime')
	fake=fltarr(1024,1024)
	fake[505:519,505:519]=flux*psf ; centered at 6.5, 6.5

	; here is where I need the logic to add the 
	     ; fake object
	imsat[*,*,i]=(imb[*,*,i]+xyshift(fake,djx-512.5+satx/kpp[i],djy-512.5+saty/kpp[i],/cubic))/rat
	imbs[*,*,i]=imb[*,*,i]/dj[i]
	imsats[*,*,i]=imsat[*,*,i]/dj[i]
endfor
psf=median(imbs,dim=3)
psfsat=median(imsats,dim=3)
for i=0,9 do imbs[*,*,i]-=psf
for i=0,9 do imsats[*,*,i]-=psfsat

for i=0,9 do imbs[*,*,i]=imbs[*,*,i]*dj[i]
for i=0,9 do imsats[*,*,i]=imsats[*,*,i]*dj[i]
; imms is the median subtracted image

; imb should never change. it's the raw data


; imsat is in raw units

imz=imsat
imzs=imsats

; imz is going to be in units of the brightness of the first image
for i=0,9 do begin
	imzz=xyshift(imsat[*,*,i],512-479,512-554)/dj[i]*dj[0]
	sz=1024*rr[i]/rr[-1]
	imzz=congrid(imzz,sz,sz)
	imzz=imzz[sz/2-512:sz/2+511,sz/2-512:sz/2+511]
	imz[*,*,i]=imzz

	imzz=xyshift(imsats[*,*,i],512-479,512-554)/dj[i]*dj[0]
	imzz=congrid(imzz,sz,sz)
	imzz=imzz[sz/2-512:sz/2+511,sz/2-512:sz/2+511]
	imzs[*,*,i]=imzz
endfor

; scale of imz is the last image.
print,'imz scale: ',kpp[-1]

end

