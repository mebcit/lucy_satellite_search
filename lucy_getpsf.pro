pro lucy_getpsf,im,plot=plot,g1,g2,ang,resid,mresid,psf=psf

; g1,g2,ang are the major and minor sigmas and the angle

; using a heigh min of 3.5 on FIND appears to get most of the visible
; real stars
; keyword psf just return a 15x15 psf centered at 6.5,6.5



sky,im,skymode,/silent
im=im-skymode
find,im,x,y,flux,sharp,roundness,3.5,2.,[-1.5,-.5],[.4,1.],/silent
ns=n_elements(x)

; get rid of blends
good=intarr(ns)+1
for i=0,ns-2 do begin
	d=sqrt((x[i]-x[i+1:-1])^2+(y[i]-y[i+1:-1])^2)
	if min(d) lt 10 then good[i]=0
endfor
w=where(good eq 1 and x gt 7 and x lt 1023-7 and y gt 7 and y lt 1023-7)
x=x[w]
y=y[w]
flux=flux[w]

f=flux(sort(flux))
w=where(flux gt f[-21]) ; 20 brightest objects


; take 20 brightest objects

if keyword_set(plot) then begin
	window,0,xs=1024,ys=1024
	loadct,0
	tvscl,(im-skymode)>(-1)<3
	circle,color=3
	loadct,3 &  tek_color & plot,xs=1,ys=1,xmargin=[0,0],ymargin=[0,0],xr=[0,1023],yr=[0,1023],x,y,ps=8,/noerase,syms=3
	circle,color=6
	oplot,x[w],y[w],ps=8,syms=4
endif

g1=fltarr(n_elements(w))
g2=g1
ang=g1
residuals=fltarr(75,75,n_elements(w))
if keyword_set(plot) then window,0,xs=1500,ys=1200
loadct,3
for i=0,n_elements(w)-1 do begin
	xoffset=i mod 5
	yoffset=fix(i/5)
	xc=fix(x[w[i]]+.5)
	yc=fix(y[w[i]]+.5)
	pim=im[xc-7:xc+7,yc-7:yc+7]
	a=[skymode,pim[5,5],1.6,1.,7.,7.,-.36]
	pfit=gauss2dfit(pim,a,/tilt,fita=[0,1,1,1,1,1,1])
	if a[6] lt !pi/2 then a[6]+=!pi
	if a[6] gt !pi/2 then a[6]-=!pi
	g1[i]=a[2]
	g2[i]=a[3]
	ang[i]=a[6]
	xc=a[4]
	yc=a[5]
	if keyword_set(plot) then print,a[2],a[3],a[6]
	;print,i,rms((pim-pfit)/a[1])
endfor
g1=median(g1)
g2=median(g2)
ang=median(ang)
; now redo the residual map using the forced gaussian
for i=0,n_elements(w)-1 do begin
	xoffset=i mod 5
	yoffset=fix(i/5)
	xc=fix(x[w[i]]+.5)
	yc=fix(y[w[i]]+.5)
	pim=im[xc-7:xc+7,yc-7:yc+7]
	a=[skymode,pim[7,7],g1,g2,7.,7.,ang]
	pfit=gauss2dfit(pim,a,/tilt,fita=[0,1,0,0,1,1,0])
	xc=a[4]
	yc=a[5]
	residuals[*,*,i]=xyshift(rebin(pim-pfit,75,75,/sample),37.-xc*5,37.-yc*5,/cubic)/a[1]
	;residuals[*,*,i]=rebin(xyshift(pim-pfit,7-xc,7-yc,/cubic),75,75,/sample)/a[1]
	if keyword_set(plot) then tvscl,rebin(residuals[*,*,i],300,300,/sample)/a[1]>(-.1)<.1,xoffset*300,yoffset*300
endfor
resid=median(residuals,dim=3) ; this is a 5x supersampled residual map. 
; now show the subtraction of the residual map
print,'PSF: ',g1,g2,ang
mresid=fltarr(75,75,n_elements(w)) ; even more residuals
if keyword_set(plot) then begin
	stop
	for i=0,n_elements(w)-1 do begin
		xoffset=i mod 5
		yoffset=fix(i/5)
		xc=fix(x[w[i]]+.5)
		yc=fix(y[w[i]]+.5)
		pim=im[xc-7:xc+7,yc-7:yc+7]
		a=[skymode,pim[5,5],g1,g2,7.,7.,ang]
		pfit=gauss2dfit(pim,a,/tilt,fita=[0,1,0,0,1,1,0]) ; fix the 2d gaussian
		xc=a[4]
		yc=a[5]
		mresid[*,*,i]=xyshift(rebin(pim-pfit,75,75,/sample),37.-xc*5,37.-yc*5,/cubic)/a[1]-resid
		tvscl,rebin(mresid[*,*,i]>(-.1)<.1,300,300),xoffset*300,yoffset*300
		;print,i,rms(mresid[*,*,i])
	endfor
endif
psf=pim/total(pim)
return
;save,g1,g2,ang,resid,fi='psf_residuals.sav'
end




