import pygame
from pygame.locals import *
import colorsys
import random
import sounddevice as sd
from scipy.io.wavfile import write
import pywt
import librosa
import numpy as np

import math

w, h = 720, 720

nPoints = 500

pygame.init()
screen = pygame.display.set_mode((w, h), RESIZABLE)
pygame.display.set_caption("Cochlea Visualization")
clock = pygame.time.Clock()

# todo: save score and best score, add start screen

# graphics
#cochlea_img = pygame.image.load("cochlea_1.png")
#cochlea_img_size = cochlea_img.get_rect().size

#font = pygame.font.Font("custom_font (2).ttf", 40)


space_was_pressed = False


def init_parameters():
	pass


def draw_img(img, img_x, img_y, img_w, img_h):
	screen.blit(pygame.transform.scale(img, (img_w, img_h)), (img_x, img_y))

def clamp(v, minV, maxV):
	if v < minV:
		return minV
	if v > maxV:
		return maxV
	return v

def wav_to_bins(filename, num_bins):
    # Load the audio file with librosa
    samples, sample_rate = librosa.load(filename, sr=None)

    # For CWT
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(samples, scales, 'morl', 1.0 / sample_rate)

    # Define logarithmically spaced frequency bins
    bin_edges = np.logspace(np.log10(frequencies[1]), np.log10(frequencies[-1]), num_bins + 1)

    # Initialize an array to hold the total amplitude for each bin
    total_amplitudes = np.zeros(num_bins)

    # Sum the amplitudes within each bin
    for i in range(num_bins):
        x = zip(coefficients, frequencies)
        for pair in x:
            if bin_edges[i+1] <= pair[1] < bin_edges[i]:
                total_amplitudes[i] += np.sum(abs(pair[0]))
                
                
    return bin_edges, total_amplitudes


def renderCochlea(bins, w, h):
	def hsv_color_function(i):
		ratio = i/nPoints
		normalizedIntensity = bins[math.floor(ratio*len(bins))]
		#preciseBinPosition = (len(bins)-1/2)*ratio
		#binBeforeIndex = clamp(math.floor(preciseBinPosition), 0, len(bins)-1)
		#binAfterIndex = clamp(math.ceil(preciseBinPosition), 0, len(bins)-1)
		#normalizedIntensity = bins[binBeforeIndex]*(preciseBinPosition-binBeforeIndex)+bins[binAfterIndex]*(binAfterIndex-preciseBinPosition)
		return colorsys.hsv_to_rgb((1-normalizedIntensity)*2/3,1,255)
	def draw_pattern(colorFunction, size):
		for i in range(nPoints):
			r = i/nPoints*minDim/2*0.75
			theta = math.pi+i/nPoints*4*math.pi
			x = int(w/2+math.cos(theta)*r)
			y = int(h/2+math.sin(theta)*r)
			r, g, b = colorFunction(i)
			#print(color)
			pygame.draw.circle(screen, (r, g, b), (x, y), size)

	def white_color_function(i):
		return (255, 255, 255)
	def black_color_function(i):
		return (0, 0, 0)
	screen.fill(pygame.Color(0, 0, 0))

	minDim = min(w, h)
	maxDim = max(w, h)

	# todo: add infinite sky
	#rendered_cochlea_min_dim = int(h*cochlea_img_size[0]/cochlea_img_size[1])
	#rendered_cochlea_min_dim = int(h*cochlea_img_size[0]/cochlea_img_size[1])
	#draw_img(cochlea_img, w/2-cochlea_img_size[0]/2, h/2-cochlea_img_size[1]/2, int(h*cochlea_img_size[0]/cochlea_img_size[1]), h)

	w, h = pygame.display.get_surface().get_size()

	big_circle_r = minDim/20
	small_circle_r = minDim/25
	draw_pattern(white_color_function, big_circle_r)
	draw_pattern(black_color_function, small_circle_r)
	draw_pattern(hsv_color_function, small_circle_r)

	# pygame.draw.rect(screen, "yellow", pygame.Rect(int(bird_x*w), int(bird_y*h), int(bird_w*h), int(bird_h*h)))

	#font_img = font.render(str(score)+"/"+str(high_score), True, "white")
	#screen.blit(font_img, (w/64, h/64))

def runLoop():
	init_parameters()
	running = True
	counter = 0
	while running:
		filename = 'output.wav'
		sample_seconds = 1  # Duration of recording
		sample_rate = 44100  # Sample rate
		nsamples = 1
		for i in range(nsamples):
			recording = sd.rec(int(sample_seconds * sample_rate), samplerate=sample_rate, channels=2)
			sd.wait() # Wait until recording is finished
			write(filename, sample_rate, recording)  # Save as WAV file
		bins = wav_to_bins("C:\\Users\\Adam\\Desktop\\personal projects\\0_mcgill_hackathon_2023\\Simulated-Auditory-Evoked-Hemodynamics\\"+filename, 64)[1]
		bins = bins/max(bins)


		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False

		renderCochlea(bins, w, h)

		pygame.display.flip()

		"""
		if pygame.key.get_pressed()[pygame.K_SPACE]:
			if not space_was_pressed and bird_v > minimum_v_to_jump:
				bird_v = bird_jump_v
				space_was_pressed = True
		else:
			space_was_pressed = False
		"""

		clock.tick(int(1/(sample_seconds+1)))
		counter += 1


	pygame.quit()
