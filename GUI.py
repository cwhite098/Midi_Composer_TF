import pygame
import numpy as np
import pretty_midi as pm
import cv2

midi_data = pm.PrettyMIDI('test.mid')
piano_roll = midi_data.get_piano_roll(10)
print(piano_roll.shape)


def paddedzoom2(img, zoomfactor=0.8):
    
    ' does the same thing as paddedzoom '
    
    h,w = img.shape
    M = cv2.getRotationMatrix2D( (w/2,h/2), 0, zoomfactor) 
    
    return cv2.warpAffine(img, M, img.shape[::-1])



def run_GUI(piano_roll, size_x, size_y):
    pygame.init()
    display = pygame.display.set_mode((size_x, size_y))
    print(piano_roll.shape)
    
    surf = pygame.surfarray.make_surface(piano_roll)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False


        display.blit(surf, (0,0))
        pygame.display.update()


    pygame.quit()

piano_roll[piano_roll!=0] = 1
piano_roll = paddedzoom2(piano_roll, 3)

run_GUI(piano_roll, piano_roll.shape[0], piano_roll.shape[1])