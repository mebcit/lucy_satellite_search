diametersat=2.5
albedosat=0.41
satdist=70.+randomu(iseed)*1
satdist=15.
satang=randomu(iseed)*360.
satx=satdist*cos(satang*!dtor)
saty=satdist*sin(satang*!dtor)

; compare straight pairs (or more)
fs=file_search('../llori/2025110/*1x1*fit')

wlist=list()
wlist.add,[165,167]
;wlist.add,[176,178]
;wlist.add,[180,182]
;wlist.add,[184,186]
;wlist.add,[188,190]
;wlist.add,[199,201]
;wlist.add,[217,219]
;wlist.add,[238,239,240,241,242,243,244,245]
;wlist.add,[276,277,278]
;wlist.add,[282,283,284,285]
;wlist.add,[270,288]


cx=cos(findgen(361)*!dtor)
cy=sin(findgen(361)*!dtor)

im=readfits(fs[100])
lucy_getpsf,im,psf=psf
for j=0,n_elements(wlist)-1 do begin
	wf=wlist[j]
	imp=fltarr(1024,1024,n_elements(wf))
	for i=0,n_elements(wf)-1 do begin
		imp[*,*,i]=readfits(fs[wf[i]],h,/silent)
		sky,imp[*,*,i],m,skysig,/sky
		imp[*,*,i]=imp[*,*,i]-m
		; add a fake satellite!
		getgeometry,h,range,phase,delta
		lucy_finddj,h,xpred,ypred
		flux=fakesat(diametersat,albedosat,range,delta,phase)*sxpar(h,'exptime')
		fake=fltarr(1024,1024)
		fake[505:519,505:519]=flux*psf>0 ; centered at 6.5, 6.5
		kpp=range*!dtor/3600.
		mx=max(total(imp[*,*,i],2),dx)
		mx=max(total(imp[*,*,i],1),dy)

		imp[*,*,i]=imp[*,*,i]+xyshift(fake,dx-512.5+satx/kpp,dy-512.5+saty/kpp,/cubic)
	
	
		jd=sxpar(h,'midutcjd')
		loadct,0
		tvscl,imp[*,*,i]>(-skysig*4)<(skysig*5.)
	
		jd=double(strmid(jd,3,30))
		extast,h,ast
		refcat,ast.crval[0],ast.crval[1],.3,.3,star
		et=sxpar(h,'exptime')
		print,'exposure time:',et
		w=where(star.r lt 15 and star.r gt 5,c)
		star=star[w]
		ad2xy,star.ra,star.dec,ast,x,y
		;x=x-24
		;y=y+67
		x=x+dx-xpred
		y=y+dy-ypred
		tek_color
		plot,[0,1],/nodata,xr=[0,1024],yr=[0,1024],xs=5,ys=5,xmargin=[0,0],ymargin=[0,0],/noerase
		circle,color=2
		oplot,x,y,ps=8,syms=3,thick=2
		xyouts,x,y+20,strmid(strcompress(star.r,/re),0,4),color=2,charsize=3,charthick=3,align=.5
		rrr=[20,30,40,50,60,70,80]
		for iii=0,n_elements(rrr)-1 do begin
			oplot,dx+cx*rrr[iii]/kpp,dy+cy*rrr[iii]/kpp,color=2,thick=1
		endfor

		print,'   scale (km pp):',1./3600*!dtor*range
		print,'time to CA:',(jd-2460786.2439811)*24,'WRONG'
		print,'Add to mag: ',alog10(1./((range/1.5e8)^2*(delta/1.5e8)^2))/2*5
		print,'max sep:',min([dx,dy,1024-dx,dy])*kpp
		flux=fakesat(1.,albedosat,range,delta,phase)*sxpar(h,'exptime')
		smallim=imp(900:1003,*,0)
		sky,smallim,mm,skysig,/silent
		print,'5sig radius: ',sqrt(skysig*5*sqrt(!pi)/.93*5/flux)
		print,wf
		; that's 5 sigma in a 5 pixel radius, with a .93 aperture correction
		
	endfor
	stop
endfor

end

