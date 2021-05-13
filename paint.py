import pygame
import numpy as np
import tensorflow as tf
from PIL import Image

pygame.init()
win = pygame.display.set_mode((500,500))
win.fill((0,0,0))
clock = pygame.time.Clock()
run = True
draw_on = False

classifier = tf.keras.models.load_model('number-model.h5')
classifier.summary()

while run:
	#win.fill((255,255,255))

	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False

		keys = pygame.key.get_pressed()

		if event.type == pygame.MOUSEBUTTONDOWN:
			x, y = pygame.mouse.get_pos()
			pygame.draw.rect(win, (255,255,255), (x,y,60,60))
			draw_on = True

		if event.type == pygame.MOUSEMOTION:
			x, y = pygame.mouse.get_pos()
			if draw_on:
				pygame.draw.rect(win, (255,255,255), (x,y,60,60))

		if event.type == pygame.MOUSEBUTTONUP:
			draw_on = False

		if keys[pygame.K_RETURN]:
			pygame.image.save(win, 'screenshot.png')
			img = Image.open('screenshot.png')
			img = img.resize((28, 28))
			img.save('screenshot.png')
			im = np.array(Image.open('screenshot.png').convert('L'))
			im.shape = (1,28,28)
			probability_model = tf.keras.Sequential([classifier, tf.keras.layers.Softmax()])
			predictions = probability_model.predict(im)
			predictions = np.argmax(predictions)
			print(predictions)

		if keys[pygame.K_c]:
			win.fill((0,0,0))


	pygame.display.update()
	clock.tick(60)

