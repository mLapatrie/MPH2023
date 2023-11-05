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


from pydub import AudioSegment


bin_edge = np.load('bin_edges.npy')


w, h = 350, 350
A = 165.4
a = 2.1
k = 0.88
t_m = 5 * math.pi #theta max
total_arc = (np.arcsinh(t_m) + t_m * np.sqrt((t_m) ** 2 + 1))/2

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

images_saved = 0
counter = 0
#space_was_pressed = False


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

def frequency_i(i):
	ratio = i/nPoints
	theta_i = ratio * 5 * math.pi
	frequency = A * 10 ** (a/(2 * total_arc) * (np.arcsinh(theta_i) + theta_i * np.sqrt((theta_i) ** 2 + 1))) - k
	return frequency

def bin_i(i):

	for index in range(1, len(bin_edge)):
		if frequency_i(i) > 6500:
			return 0
		if frequency_i(i) <= bin_edge[index - 1] and frequency_i(i) >= bin_edge[index]:
			return index



def renderCochlea(bins, w, h):
	def hsv_color_function(i):
		#normalizedIntensity = bins[math.floor(ratio*len(bins))]
		normalizedIntensity = bins[bin_i(i)]
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

def saveImage():
	global images_saved
	pygame.image.save(screen, f"cochlea_frames\\frame_{images_saved:04d}.png")
	images_saved += 1

def runLoop():
	global counter
	init_parameters()
	running = True
	while running:
		input_filename = 'intro_silence.wav'
		output_filename = "output.wav"
		newAudio = AudioSegment.from_wav(input_filename)
		newAudio = newAudio[counter*100:(counter+1)*100] # 100 milliseconds for 10 frames per second
		newAudio.export(output_filename, format="wav")  # Exports to a wav file in the current path.

		bins = wav_to_bins(output_filename, 127)[1]
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

		clock.tick(60)
		counter += 1
		saveImage()


	pygame.quit()

runLoop()