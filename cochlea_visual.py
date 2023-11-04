import pygame
from pygame.locals import *

import math

w, h = 1280, 720

angularRatio = 0.001
nPoints = 160

pygame.init()
screen = pygame.display.set_mode((w, h), RESIZABLE)
pygame.display.set_caption("Cochlea Visualization")
clock = pygame.time.Clock()

# todo: save score and best score, add start screen

# graphics
#sky_img = pygame.image.load("sky.png")
#sky_img_size = sky_img.get_rect().size

#font = pygame.font.Font("custom_font (2).ttf", 40)


space_was_pressed = False


def init_parameters():
	pass


def draw_img(img, img_x, img_y, img_w, img_h):
	screen.blit(pygame.transform.scale(img, (img_w, img_h)), (img_x, img_y))

init_parameters()

running = True
while running:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	screen.fill(pygame.Color(0, 0, 0))

	# todo: add infinite sky

	#draw_img(sky_img, 0, 0, int(h*sky_img_size[0]/sky_img_size[1]), h)

	w, h = pygame.display.get_surface().get_size()

	minDim = min(w, h)
	for i in range(nPoints):
		x = w/2+math.cos(i/10)*i
		y = h/2+math.sin(i/10)*i
		pygame.draw.circle(screen, (255, 0, 0), (x, y), 5)

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


pygame.quit()