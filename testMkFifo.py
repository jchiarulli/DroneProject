import cv2
import subprocess as sp
import numpy

FFMPEG_BIN = "ffmpeg"
command = [ FFMPEG_BIN,
        '-i', 'fifo264',             # fifo is the named pipe
        '-pix_fmt', 'bgr24',      # opencv requires bgr24 pixel format.
        '-vcodec', 'rawvideo',
        '-an','-sn',              # we want to disable audio processing (there is no audio)
        '-f', 'image2pipe', '-']    
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

while True:
    # Capture frame-by-frame
    raw_image = pipe.stdout.read(640*480*3)
    # transform the byte read into a numpy array
    image =  numpy.fromstring(raw_image, dtype='uint8')
    image = image.reshape((480,640,3))          # Notice how height is specified first and then width
    if image is not None:
        cv2.imshow('Video', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    pipe.stdout.flush()

cv2.destroyAllWindows()
