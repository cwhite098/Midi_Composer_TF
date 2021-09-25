
from bs4 import BeautifulSoup
from urllib.request import urlopen, urlretrieve
import time
from music21 import converter, instrument, stream
import os


def Scrape(dir):
## This Function scrapes Mutopia website for Jazz Piano Midi Files.
## Change the URL to get different instruments and styles.

    # Directory to save the Midi files
    save_dir = dir

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
        print('Making Directory:' + str(save_dir))

    # Url is plsit into two so we can go through the pages of the search results
    url = 'https://bushgrafts.com/midi/'

    # Init values
    song_number = 0
    link_count = 10

    file_name = 0

    # main loop

    #finds the correct page of search results
    html = urlopen(url)
    soup = BeautifulSoup(html.read())
    # Finds all the links on the page
    links = soup.find_all('a')
    link_count = 0

    for link in links:
        if link.has_attr('href'):
            href = link['href']
            # Find all links with a .mid in them
            if href.find('.mid') >= 0:
                link_count = link_count + 1
                #Download that link
                urlretrieve(href, dir+str(file_name)+'.mid' )
                file_name += 1

    #+10 since there are 10 results on each page
    song_number += 10
    # Small wait to be nice to the website
    time.sleep(10.0)


def extract_piano(midi_file, name, location):

    midi = converter.parse(midi_file)
    parts = midi.parts.stream()

    

    for part in parts:
        print(part.partName)

        if part.partName == None:
            print('uh oh')
            continue
        elif 'Piano' in part.partName or 'piano' in part.partName:
            #midi_stream = stream.Stream(part)
            print(' saving: piano_only/'+str(name)+'.mid')
            part.write('midi', fp='piano_only/'+str(name))


def main():

    Scrape('piano2/')

    song_list = os.listdir('piano2/')
    
    location = 'piano_only/'

    if not os.path.isdir(location):
        os.mkdir(location)
        print('Making Directory:' + str(location))

    for song in song_list:
        extract_piano('piano2/'+str(song),song, location)

if __name__ == '__main__':
    main()
