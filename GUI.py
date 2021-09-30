import pygame
import numpy as np
import pretty_midi as pm
import cv2
from pygame import mouse
import scipy.ndimage
from mido import MidiFile
import pickle
from Composer import load_list, prep_data, make_or_restore, generate_seq, generate_midi
import shutil, os

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 50, 50)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 50)
BLUE = (50, 50, 255)
GREY = (200, 200, 200)
ORANGE = (200, 100, 50)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
TRANS = (1, 1, 1)


class button():
    # https://www.youtube.com/watch?v=4_9twnEduFA
    def __init__(self, color, x, y, width, height, text='', shape='square'):
        self.color = color
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.shape = shape

    def draw(self,win,outline=None):
        #Call this method to draw the button on the screen
        if outline:
            pygame.draw.rect(win, outline, (self.x-2,self.y-2,self.width+4,self.height+4),0)
        if self.shape == 'square':
            pygame.draw.rect(win, self.color, (self.x,self.y,self.width,self.height),0)
        if self.shape == 'dtriangle':
            pygame.draw.polygon(win, self.color, [(self.x, self.y), (self.x+self.width, self.y), (self.x+(self.width/2), self.y+self.height)] )
        if self.shape == 'utriangle':
            pygame.draw.polygon(win, self.color, [(self.x, self.y+self.height), (self.x+self.width, self.y+self.height), (self.x+(self.width/2), self.y)] )
        
        if self.text != '':
            font = pygame.font.SysFont('comicsans', 20)
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
    piano_roll = np.flipud(piano_roll)

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

    input_chords, input_dur, target_chords, target_dur, int_to_chord, int_to_dur, no_chords, no_dur = prep_data()
    

    model, initial_epoch = make_or_restore('./weights', input_chords, no_chords, input_dur, no_dur)
    #train(model, input_chords, input_dur, target_chords, target_dur, initial_epoch)
    
    
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
    roll_display = pygame.Surface((size_x, 260))
    
    surf = pygame.surfarray.make_surface(piano_roll)

    roll_speed = 0
    
    running = True
    y_coord = 0

    play_button = button((100,100,100), 20, 300, 100, 50,'Play')
    pause_button = button((100,100,100), 220, 300, 100, 50,'Pause')
    rewind_button = button((100,100,100), 420, 300, 100, 50,'Rewind')

    gen_button = button((100,100,100), 220, 450, 100, 50, 'Generate Song')
    save_button = button((100,100,100), 220, 600, 100, 50, 'Save Song')

    vol = 50
    vol_up = button((100,100,100), 45, 450-43, 50, 43, '', 'utriangle')
    vol_down = button((100,100,100), 45, 500, 50, 43, '', 'dtriangle')
    temp = 50
    temp_up = button((100,100,100), 445, 450-43, 50, 43, '', 'utriangle')
    temp_down = button((100,100,100), 445, 500, 50, 43, '', 'dtriangle')
    font = pygame.font.SysFont('comicsans', 24)
    

    buttons = [play_button, pause_button, rewind_button, gen_button, save_button, vol_up, vol_down, temp_up, temp_down]

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
                if rewind_button.isOver(mouse_pos):
                    pygame.mixer.music.stop()
                    pygame.mixer.music.rewind()
                    roll_speed = 0
                    y_coord = 0       
                if vol_up.isOver(mouse_pos):
                    vol += 10
                    if vol>100:
                        vol=100
                    pygame.mixer.music.set_volume(vol/100)
                if vol_down.isOver(mouse_pos):
                    vol -= 10
                    if vol<0:
                        vol=0
                if temp_up.isOver(mouse_pos):
                    temp += 10
                    if temp>100:
                        temp=100

                if temp_down.isOver(mouse_pos):
                    temp -= 10
                    if temp<0:
                        temp=0

                if save_button.isOver(mouse_pos):
                    if not os.path.isdir('Saved_Songs'):
                        os.mkdir('Saved_Songs')
                        print('Making Directory: Saved_Songs')
                    song_name = input('Enter song name: ')
                    shutil.copy('test.mid', 'Saved_Songs/'+str(song_name)+'.mid')

                if gen_button.isOver(mouse_pos):
                    pygame.mixer.music.stop()

                    start_chords = np.random.randint(0, len(input_chords)-1)
                    start_dur = np.random.randint(0, len(input_dur) -1)

                    chord_pattern = input_chords[start_chords]
                    dur_pattern = input_dur[start_dur]
                    
                    print('generating notes...')
                    predicted_notes = generate_seq(model, chord_pattern, dur_pattern, int_to_chord, int_to_dur, temp/100)

                    generate_midi(predicted_notes, 'test.mid')

                    pygame.mixer.music.load("test.mid")
    
                    mid = MidiFile('test.mid')
                    max_time = mid.length

                    piano_roll = process_midi('test.mid')
    
                    surf = pygame.surfarray.make_surface(piano_roll)
                    size_x = piano_roll.shape[0]
                    size_y = piano_roll.shape[1]
                    roll_display = pygame.Surface((size_x, 260))
                    y_coord = 0
                    
                    roll_speed = 0
              
            if event.type == pygame.MOUSEMOTION:
                for b in buttons:
                    if b.isOver(mouse_pos):
                        b.color = (150,150,150)
                    else:
                        b.color = (100,100,100)


        display.fill(WHITE)

        display.blit(roll_display,((540-size_x)/2 ,20 ))
        roll_display.blit(surf, ((540-size_x)/2,y_coord-size_y+260))
        y_coord += roll_speed

        mouse_pos = pygame.mouse.get_pos()

        for b in buttons:
            b.draw(display)
        vol_text = font.render('Vol: '+str(int(vol))+'%', 1, (0,0,0))
        display.blit(vol_text, (40 , 470))
        temp_text = font.render('Temp: '+str(int(temp))+'%', 1, (0,0,0))
        display.blit(temp_text, (435 , 470))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

def main():

    piano_roll = process_midi('test.mid')
    run_GUI(piano_roll, piano_roll.shape[0], piano_roll.shape[1])


if __name__ == '__main__':
    main()