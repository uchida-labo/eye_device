import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load('/Users/nakanokota/Downloads/SE.mp3')

while True:
    pygame.mixer.music.play()
    time.sleep(1.25)

pygame.quit()
