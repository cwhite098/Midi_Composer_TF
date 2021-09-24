import pygame
import numpy as np
import pretty_midi as pm
import cv2
import scipy.ndimage
from mido import MidiFile



def process_midi(midi_file):

    midi_data = pm.PrettyMIDI('test.mid')
    piano_roll = midi_data.get_piano_roll(10)
    print(piano_roll.shape)

    piano_roll[piano_roll!=0] = 255
    piano_roll = crop_array(piano_roll, 5)
    piano_roll = scipy.ndimage.zoom(piano_roll, 6, order=0)
    piano_roll = np.rot90(piano_roll, 2)

    return piano_roll


def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr!=0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def crop_array(array, padding):

    x_refs_l = first_nonzero(array, axis=0, invalid_val=1000)
    x_refs_r = last_nonzero(array, axis=0, invalid_val=-1)

    x_crop_l = np.amin(x_refs_l)-padding
    x_crop_r = np.amax(x_refs_r)+padding
  
    return array[x_crop_l:x_crop_r,:]


def run_GUI(piano_roll, size_x, size_y):

    pygame.init()
    clock = pygame.time.Clock()
    FPS = 30

    WHITE = [255,255,255]

    pygame.mixer.music.load("test.mid")
    
    mid = MidiFile('test.mid')
    max_time = mid.length
    print(max_time)



    display = pygame.display.set_mode((500, 700))
    display.fill(WHITE)
    roll_display = pygame.Surface((size_x, 200))
    
    surf = pygame.surfarray.make_surface(piano_roll)

    roll_speed = size_y/(max_time * FPS)
    

    running = True
    y_coord = 0

    pygame.mixer.music.play()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        
        display.blit(roll_display,((500-size_x)/2 ,20 ))
        roll_display.blit(surf, (0,y_coord-size_y+200))
        y_coord += roll_speed
        print(y_coord)


        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()




def main():

    piano_roll = process_midi('test.mid')
    run_GUI(piano_roll, piano_roll.shape[0], piano_roll.shape[1])


if __name__ == '__main__':
    main()