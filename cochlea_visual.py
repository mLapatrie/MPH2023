import pygame
from pygame.locals import *
import colorsys
import random

import math

w, h = 1280, 720

#test variable
a = 0

bins = [0.25, 0.5, 0.75] #[random.random() for i in range(16)]
bins = [random.random() for i in range(16)]
print(bins)

angularRatio = 0.001
nPoints = 500

pygame.init()
screen = pygame.display.set_mode((w, h), RESIZABLE)
pygame.display.set_caption("Cochlea Visualization")
clock = pygame.time.Clock()

# todo: save score and best score, add start screen

# graphics
cochlea_img = pygame.image.load("cochlea_1.png")
cochlea_img_size = cochlea_img.get_rect().size

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

init_parameters()

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	screen.fill(pygame.Color(0, 0, 0))

	minDim = min(w, h)
	maxDim = max(w, h)

	# todo: add infinite sky
	rendered_cochlea_min_dim = int(h*cochlea_img_size[0]/cochlea_img_size[1])
	rendered_cochlea_min_dim = int(h*cochlea_img_size[0]/cochlea_img_size[1])
	draw_img(cochlea_img, w/2-cochlea_img_size[0]/2, h/2-cochlea_img_size[1]/2, int(h*cochlea_img_size[0]/cochlea_img_size[1]), h)

	w, h = pygame.display.get_surface().get_size()



	def draw_pattern(colorFunction, size):
		for i in range(nPoints):
			r = i/nPoints*minDim/2*0.75
			theta = math.pi+i/nPoints*4*math.pi
			x = int(w/2+math.cos(theta)*r)
			y = int(h/2+math.sin(theta)*r)
			color = colorFunction(i)
			#print(color)
			pygame.draw.circle(screen, color, (x, y), size)

	def white_color_function(i):
		return (255, 255, 255)
	def black_color_function(i):
		return (0, 0, 0)
	def hsv_color_function(i):
		ratio = i/nPoints
		preciseBinPosition = (len(bins)-1/2)*ratio
		binBeforeIndex = clamp(math.floor(preciseBinPosition), 0, len(bins)-1)
		binAfterIndex = clamp(math.ceil(preciseBinPosition), 0, len(bins)-1)
		normalizedIntensity = bins[binBeforeIndex]*(preciseBinPosition-binBeforeIndex)+bins[binAfterIndex]*(binAfterIndex-preciseBinPosition)
		return colorsys.hsv_to_rgb((1-normalizedIntensity)*2/3,1,255)
			
	draw_pattern(white_color_function, 35)
	draw_pattern(black_color_function, 25)
	draw_pattern(hsv_color_function, 25)

	# pygame.draw.rect(screen, "yellow", pygame.Rect(int(bird_x*w), int(bird_y*h), int(bird_w*h), int(bird_h*h)))

	#font_img = font.render(str(score)+"/"+str(high_score), True, "white")
	#screen.blit(font_img, (w/64, h/64))

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
	a += 0.0001
	#print(a)


pygame.quit()