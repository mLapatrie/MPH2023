import pyaudio
import numpy as np
import time

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
               channels=1,
               rate=44100,
               input=True,
               frames_per_buffer=1024)

while True:
   data = stream.read(1024)
   np_data = np.fromstring(data, dtype=np.int16)
   rms = np.sqrt(np.mean(np_data**2))
   print(rms)
   time.sleep(1)

stream.stop_stream()
stream.close()
p.terminate()