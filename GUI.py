import pygame
import numpy as np
import pretty_midi as pm
import cv2
import scipy.ndimage
from mido import MidiFile


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


class Slider():

# Use https://github.com/HackerPoet/FaceEditor/blob/master/face_edit.py instead for sliders

    def __init__(self, name, val, max, min, pos):
        font = pygame.font.SysFont("comicsans", 12)
        self.val = val
        self.max = max
        self.min = min
        self.x = pos[0]
        self.y = pos[1]
        self.surf = pygame.surface.Surface((100,50))
        self.hit = False
        self.txt_surf = font.render(name, 1, (0,0,0))
        self.txt_rect = self.txt_surf.get_rect(center=(50,15))

        #Background surface that doesnt move
        self.surf.fill((100,100,100))
        pygame.draw.rect(self.surf, GREY, [0,0,100,50], 3)
        pygame.draw.rect(self.surf, ORANGE, [10, 10, 80, 10], 0)
        pygame.draw.rect(self.surf, WHITE, [10, 30, 80, 5], 0)

        self.surf.blit(self.txt_surf, self.txt_rect)

        # Surface for button that moves
        self.button_surf = pygame.surface.Surface((20, 20))
        self.button_surf.fill(TRANS)
        self.button_surf.set_colorkey(TRANS)
        pygame.draw.circle(self.button_surf, BLACK, (10, 10), 6, 0)
        pygame.draw.circle(self.button_surf, ORANGE, (10, 10), 4, 0)

    def draw(self, screen):
        surf = self.surf.copy()

        pos = (10+int((self.val-self.min)/(self.max-self.min)*80), 33)
        self.button_rect = self.button_surf.get_rect(center=pos)
        surf.blit(self.button_surf, self.button_rect)
        self.button_rect.move_ip(self.x, self.y)  # move of button box to correct screen position

        screen.blit(surf, (self.x, self.y))

    def move(self):
        if self.hit == True:
            self.val = (pygame.mouse.get_pos()[0] - self.x - 10) / 80 * (self.max - self.min) + self.min
            pygame.mixer.music.set_volume(self.val)
            print(self.val)
            if self.val < self.min:
                self.val = self.min
            if self.val > self.max:
                self.val = self.max



def process_midi(midi_file):

    midi_data = pm.PrettyMIDI('test.mid')
    piano_roll = midi_data.get_piano_roll(10)
    print(piano_roll.shape)

    piano_roll[piano_roll!=0] = 255
    piano_roll = crop_array(piano_roll, 5)
    piano_roll = scipy.ndimage.zoom(piano_roll, 6, order=0)
    piano_roll = np.rot90(piano_roll, 2)
    pinao_roll = np.fliplr(piano_roll)

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

    music = pygame.mixer.music.load("test.mid")
    
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

    play_button = button((100,100,100), 20, 300, 100, 50,'Play')
    pause_button = button((100,100,100), 220, 300, 100, 50,'Pause')
    rewind_button = button((100,100,100), 420, 300, 100, 50,'Rewind')

    vol_slider = Slider('Volume', 1, 1, 0, (20, 400))
    slides = [vol_slider]
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
                for s in slides:
                    if s.button_rect.collidepoint(mouse_pos):
                        s.hit = True

            if event.type == pygame.MOUSEBUTTONUP:
                for s in slides:
                    s.hit = False
   
            if event.type == pygame.MOUSEMOTION:
                if play_button.isOver(mouse_pos):
                    play_button.color = (150,150,150)
                else:
                    play_button.color = (100,100,100)
                if pause_button.isOver(mouse_pos):
                    pause_button.color = (150,150,150)
                else:
                    pause_button.color = (100,100,100)
                if rewind_button.isOver(mouse_pos):
                    rewind_button.color = (150,150,150)
                else:
                    rewind_button.color = (100,100,100)

        display.blit(roll_display,((540-size_x)/2 ,20 ))
        roll_display.blit(surf, ((540-size_x)/2,y_coord-size_y+200))
        y_coord += roll_speed

        mouse_pos = pygame.mouse.get_pos()

        play_button.draw(display)
        pause_button.draw(display)
        rewind_button.draw(display)
        
        for s in slides:
            s.move()
            s.draw(display)

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()




def main():

    piano_roll = process_midi('test.mid')
    run_GUI(piano_roll, piano_roll.shape[0], piano_roll.shape[1])


if __name__ == '__main__':
    main()