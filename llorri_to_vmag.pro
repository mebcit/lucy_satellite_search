pro llorri_to_vmag,rate,vmag,ac=ac,cc=cc,stype=stype,zpt=zpt,$
	bin=bin,silent=silent,_ref_extra=extra
;+
;NAME:
;	llorri_to_vmag
;PURPOSE:
;	Convert observed L'LORRI count rates (DN/sec) into Johnson V-mag.
;CALLING SEQUENCE:
;	llorri_to_vmag,rate,vmag[,+ optional keywords]
;INPUTS:
;	rate = observed count rate in DN/sec (flux in the relevant aperture)
;OUTPUTS:
;	vmag = Johnson-Landolt V-mag
;KEYWORDS:
;	ac = aperture correction (default=0.08, which
;		is appropriate for a 5-pixel radius aperture in 1x1 format)
;	cc = color correction (default=0.0)
;	stype = "ob" for O or B stellar types,
;		"a" for A stellar types
;		"f" for F stellar types
;		"g" for G stellar types
;		"g2v" for G2V stellar types
;		"solar" for F and G stellar types
;		"k" for K stellar types
;		"m" for M stellar types
;		"red_trojan" for red Trojan type spectrum
;		"gray_trojan" for gray Trojan type spectrum
;		"didymos" when using a Didymos spectrum (S-type)
;		default is "solar"
;	zpt = zero point for mag conversion (default=18.973 for 1x1, 19.002 for 4x4)
;	bin = set this keyword for 4x4 format
;	silent = set this keyword to disable printing the output
;REVISION HISTORY:
;	HAW (11/13/2021): Original procedure.
;	HAW (02/19/2022): Updated after photometry of cal standard HD37962.
;	HAW (04/21/2023): Fixed some documentation above (STYPE descriptions).
;	HAW (05/24/2023): Added "didymos" as a possible STYPE keyword.
;						I also updated the ZPT values to be consistent with
;						the results from the Feb 2022 observations of HD37962.
;	HAW (06/01/2023): Updated CC for Didymos (changed from 0.161 to 0.054). 
;
;COMMENTS:
;	;Note that the error in magnitude is related to the error in flux by:
;		magerr = 1.0857 * (eflux/flux) = 1.0857 / SNR		;1.0857 = alog(10.)/2.5
;-

if ( n_params() lt 1 ) then begin
	print,'Must supply at least the count rate (dn/s)'
	return
endif

;Aperture correction coeffient used to convert photometry in a 5-pixel radius aperture
;to "infinite" aperture.
;Using a composite image of HD37962 taken on 2/13/2022, I found that a 5-pixel
;radius aperture has 0.919 of the total flux (after adding 0.25 DN to the raw
;numbers because the bias level was apparently oversubtracted), which implies
;AC=0.08 for 1x1 images using a 5-pixel radius aperture.

if not n_elements(ac) then ac = 0.08

if keyword_set(stype) then begin

	stype = strlowcase(stype)
	case stype of
		'ob': cc = -0.071
		'a': cc = -0.036
		'f': cc = 0.027
		'g': cc = 0.093
		'g2v': cc = 0.040
		'solar': cc = 0.0
		'k': cc = 0.39
		'm': cc = 1.20
		'red_trojan': cc = 0.028
		'gray_trojan': cc = 0.000
		'didymos': cc = 0.054
	endcase
	if not keyword_set(silent) then $
		print,"Spectral type is: ",stype
		
endif

;Assume solar color, unless specified otherwise.

if not n_elements(cc) then cc = 0.0

;****************************************************************************
;For NH-LORRI, the PASP 2021 paper gives:
;ZPT=18.78 for 1x1
;ZPT=18.88 for 4x4
;
;The above values were based on in-flight measuresments of the absolute
;standard calibration star HD37962, which has V=7.850.
;
;For that star, the NH-LORRI count rates were:
;506.41e03 e/s for 1x1 (gain=21.0) --> 24,115 DN/s
;506.31e03 e/s for 4x4 (gain=19.4) --> 26,098 DN/s
;In Feb 2022, I re-generated the HD37962 photometry for NH-LORRI and got:
;500.49e03 e/s for 1x1 (gain=21.0) --> 23,833 DN/s
;499.32e03 e/s for 4x4 (gain=19.4) --> 25,738 DN/s

;On 2/13/2022, L'LORRI measured that same star and got:
;594e03 e/s for 1x1 (gain=21.1)
;578e03 e/s for 4x4 (gain=20.0)
;594./500. = 1.188 --> LLORRI is ~19% more sensitive than NH-LORRI.

;These latter results mean (using AC=0.08 for 5 pixel radius aperture for 1x1):
;ZPT=18.97 for 1x1
;ZPT=19.07 for 4x4

;On 3/5/2022, I generated new photometry keywords using LLORRI_RESPONSIVITY to get:
;ZPT=18.85 for 1x1
;ZPT=18.88 for 4x4
;But the above keywords are ~12% different from what I expect given the difference
;in the NH-LORRI and L'LORRI responsivities. Thus, I've decided to use the
;actual measured count rate for HD37962, rather than LLORRI_RESPONSIVITY, to
;derive the ZPTs for L'LORRI. I assume the following:
;S_1x1 = 28.134e03 DN/s and S_4x4 = 28.896e03 DN/s for V=7.850 
;--> ZPT_1x1 = 2.5*alog10(28.134e03) + 7.850 = 18.973
;--> ZPT_4x4 = 2.5*alog10(28.896e03) + 7.850 = 19.002
;
;On 5/24/2023, I resolved the discrepancy between the output of LLORRI_RESPONSIVITY
;and the results from the LLORRI observations of HD37962. The problem was that
;the responsivity files weren't properly updated previously.

;On 5/27/2023, I decided to use the mean flux for 5500 +/- 50 A for HD37962 to determine
;that V=7.810 for this star. Thus, the new ZPTs are:
; ZPT_1x1 = 2.5*alog10(28.134e03) + 7.810 = 18.933
; ZPT_4x4 = 2.5*alog10(28.896e03) + 7.810 = 18.962
;****************************************************************************

if not n_elements(zpt) then begin
	if keyword_set(bin) then zpt = 18.962 else zpt = 18.933		;derived from HD37962 using V=7.810
;	if keyword_set(bin) then zpt = 19.002 else zpt = 18.973		;derived from HD37962 using V=7.850
;	if keyword_set(bin) then zpt = 18.914 else zpt = 18.973
;	if keyword_set(bin) then zpt = 18.88 else zpt = 18.85
endif

vmag = (-2.5) * alog10(rate) - ac + cc + zpt

if not keyword_set(silent) then begin
	print,format='("V-mag = ",f7.3)',vmag
endif		

end
