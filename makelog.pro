openw,1,'log.txt'
cspice_furnsh,'/home/mbrown/lucy/kernels/mk/lcy.donj.science.LATEST.tm'

fs=file_search('../llori/2025110/*1x1*fit',count=count)
h=headfits(fs[457])
sec0=sxpar(h,'midsclk')

for i=0,count-1 do begin
	h=headfits(fs[i])
	sec=sxpar(h,'midutcjd')
	time=double(strmid(sec,3,30))
	cspice_utc2et,string(time,format='(f25.7)')+' JD',et
	cspice_spkpos,'Donaldjohanson',et,'J2000','none','Lucy',pos,ltime
	sec=sxpar(h,'midsclk')

	r=sqrt(total(pos^2))


	printf,1,i,fs[i],(sec-sec0)/60.,sxpar(h,'exptime'),1/3600.*!dtor*r,r,$
		format='(i3," ",a42," ",f8.2,x,f6.3,x,f10.5,x,f10.2)'
endfor
cspice_kclear

close,1
end
