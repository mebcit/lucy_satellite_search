  width = 500

  height = 500

  fps = 20

   

  surf = SURFACE(/TEST, /BUFFER, DIMENSIONS=[width,height])
  window,0,xs=500,ys=500

   

  ; Each of the following lines produces a file in

  ; a different format.

  ;oVid = IDLffVideoWrite('video_example_file_format.webm')

  ;oVid = IDLffVideoWrite('video_example_file_format.swf')

  oVid = IDLffVideoWrite('test.mp4', FORMAT='mp4')

  ; Prints out a list of supported file formats

  PRINT, "Supported file formats: ", oVid.GetFormats()

  vidStream = oVid.AddVideoStream(width, height, fps)

   

  FOR i = 0, 10 do begin

plot,[0,i],[0,i],xr=[0,10],yr=[0,10]
    frame=tvrd(/true)

    !NULL = oVid.Put(vidStream, frame)

  ENDFOR

   

  oVid = 0

END
