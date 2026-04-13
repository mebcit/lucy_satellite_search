dlm_register,'/home/mbrown/idl/spice/icy/lib/icy.dlm'

fs=file_search('../2025110/*1x1*fit',count=count)


window,0,xs=1024,ys=1024
for i=0,count-1 do begin
	; it would be nice to add a 70 km (=4% of the Hill sphere) line
	im=readfits(fs[i],h)
	jd=sxpar(h,'midutcjd')
	jd=double(strmid(jd,3,30))
	lucy_finddj,h,xdj,ydj,r,ltime
	extast,h,ast
	;rdusno,ast.crval[0],ast.crval[1],.3,.3,xs,ys,bm,rm
	refcat,ast.crval[0],ast.crval[1],.3,.3,star
	et=sxpar(h,'exptime')
	if et gt 9 then maglim=18.
	w=where(star.r lt 16 and star.r gt 5,c)
	star=star[w]
	ad2xy,star.ra,star.dec,ast,x,y
	x=x-24
	y=y+67
	im=im-median(im)
	tvscl,im>(-10)<10
	plot,[0,1],/nodata,xr=[0,1024],yr=[0,1024],xs=5,ys=5,xmargin=[0,0],ymargin=[0,0],/noerase
	circle,color=2
	oplot,x,y,ps=8,syms=3,thick=2
	oplot,xdj+[-1,1]*100/r/!dtor*3600,[0,0]+ydj-150,color=2,thick=3
	print,fs[i],sxpar(h,'exptime'),35/!dtor/r*3600,'   scale (km pp):',1./3600*!dtor*r
	print,'time to CA:',(jd-2460786.2439811)*24
	atv,im
	stop
endfor
end









