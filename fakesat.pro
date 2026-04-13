function fakesat,diam,albedo,range,delta,phase

;diam in METERS

restore,'g1g2.sav'
g1=0.63
g2=0.18


H=-5*alog10(diam/1000.*sqrt(albedo)/1369.)
V0=H-2.5*alog10(g1*interpol(f1,ang,phase)+g2*interpol(f2,ang,phase)+(1-g1*g2)*interpol(f3,ang,phase))

v=v0+5*alog10(delta/1.496e8)+5*alog10(range/1.496e8)

zpt=18.933 ; from HalW's program

rate=10^(-(v-zpt)/2.5)


return,rate
end


