import pygame
import numpy as np
import pretty_midi as pm
import cv2
import scipy.ndimage
from mido import MidiFile


class button():
    # https://www.youtube.com/watch?v=4_9twnEduFA
    def __init__(self, color, x,y,width,height, text=''):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text

    def draw(self,win,outline=None):
        #Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x-2,self.y-2,self.width+4,self.height+4),0)
            
        pygame.draw.rect(win, self.color, (self.x,self.y,self.width,self.height),0)
        
        if self.text != '':
            font = pygame.font.SysFont('comicsans', 40)
            text = font.render(self.text, 1, (0,0,0))
            win.blit(text, (self.x + (self.width/2 - text.get_width()/2), self.y + (self.height/2 - text.get_height()/2)))

    def isOver(self, pos):
        #Pos is the mouse position or a tuple of (x,y) coordinates
        if pos[0] > self.x and pos[0] < self.x + self.width:
            if pos[1] > self.y and pos[1] < self.y + self.height:
                return True
            
        return False




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

    display = pygame.display.set_mode((540, 700))
    display.fill(WHITE)
    roll_display = pygame.Surface((size_x, 200))
    
    surf = pygame.surfarray.make_surface(piano_roll)

    roll_speed = 0
    
    running = True
    y_coord = 0

    play_button = button((100,100,100), 10, 450, 100, 50,'Play')
    pause_button = button((100,100,100), 500, 450, 100, 50,'Pause')
    mouse_pos = pygame.mouse.get_pos()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.isOver(mouse_pos):
                    pygame.mixer.music.play()
                    roll_speed = size_y/(max_time * FPS)
                if pause_button.isOver(mouse_pos):
                    pygame.mixer.music.stop()
                    roll_speed = 0
        
            if event.type == pygame.MOUSEMOTION:
                if play_button.isOver(mouse_pos):
                    play_button.color = (150,150,150)
                else:
                    play_button.color = (100,100,100)
                if pause_button.isOver(mouse_pos):
                    pause_button.color = (150,150,150)
                else:
                    pause_button.color = (100,100,100)

        display.blit(roll_display,((500-size_x)/2 ,20 ))
        roll_display.blit(surf, (0,y_coord-size_y+200))
        y_coord += roll_speed
        print(y_coord)

        mouse_pos = pygame.mouse.get_pos()

        play_button.draw(display)
        pause_button.draw(display)
        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()




def main():

    piano_roll = process_midi('test.mid')
    run_GUI(piano_roll, piano_roll.shape[0], piano_roll.shape[1])


if __name__ == '__main__':
    main()