import numpy as np

import serial

import time


waitTime = 0.1


# generate the waveform table

signalLength = 1024

t = np.linspace(0, 2*np.pi, signalLength)

song1 = [

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261,

  392, 392, 349, 349, 330, 330, 294,

  392, 392, 349, 349, 330, 330, 294,

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261]

song2 = [

  523, 523, 523, 464, 464, 464, 622,

  698, 784, 784, 784, 000, 000, 000,

  523, 523, 523, 464, 464, 464, 622,

  698, 784, 784, 784, 000, 000, 000,

  523, 523, 523, 464, 464, 464, 622,

  698, 784, 784, 784, 000, 000, 000,

  523, 523, 523, 464, 464, 464, 622,

  698, 784, 784, 784, 000, 000, 000]

song3 = [

  659, 587, 523, 587, 659, 659, 659,

  587, 587, 587, 587, 659, 659, 659,

  659, 587, 523, 587, 659, 659, 659,

  587, 587, 587, 587, 659, 784, 784,

  659, 587, 523, 587, 659, 659, 659,

  587, 587, 659, 587, 523, 523, 523]


# output formatter

formatter = lambda x: "%d" % x


# send the waveform table to K66F

serdev = '/dev/ttyACM0'

s = serial.Serial(serdev)

#print("Sending signal ...")

#print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
while True:
  line=s.read()
  i = int(line)
  if i == 0:
    for data in song1:

      s.write(bytes(formatter(data), 'UTF-8'))

      time.sleep(waitTime)
  if i == 1:
    for data in song2:

      s.write(bytes(formatter(data), 'UTF-8'))

      time.sleep(waitTime)
  if i == 2:
    for data in song3:

      s.write(bytes(formatter(data), 'UTF-8'))

      time.sleep(waitTime)
  else:
    s.write(bytes(formatter(200), 'UTF-8'))


for data in signalTable:

    s.write(bytes(formatter(data), 'UTF-8'))

    time.sleep(waitTime)  

s.close()

print("Signal sended")