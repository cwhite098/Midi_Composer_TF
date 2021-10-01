# Midi Composer Using TensorFlow

This is my second attempt at implementing a generative neural network to compose music, this time using TensorFlow. My first attempt was using PyTorch.

## Aims
My aims for this project were to implement an LSTM to generate Jazz piano music. I have attempted this before but I was not satisfied with the output music. To remedy this I have implemented a web-scraping script to download a better dataset than what I used before. I also have implemented an instrument separation function since I believe I had multiple instruments being treated as just the piano before.

I also aimed to create a GUI so that the music can be generated and interacted with. This includes an animated piano roll, similar to what you may see in typical DAW software.

## Technologies
Python including TensorFlow, Keras, Pygame, TKinter, music21, pickle, bs4 and more
Jupyter Notebooks

## The Network
Here is a diagram of the LSTM I have used. This is a modified version of the network used in [this repo:](https://github.com/jordan-bird/Keras-LSTM-Music-Generator)
![Network Diagram](https://github.com/cwhite098/Midi_Composer_TF/blob/main/model.png)

## Usage
The first time you clone the repo you must run BetterScraper.py to download the midi files, make sure your midi files are in a folder called 'piano' and then run Composer.py.
Then, in order to use the program you must run the GUI.py file. You can then generate a new song and listen to it back. A song can be saved to the Saved_Songs directory by hitting the 'Save' button and giving the file a name.
The temperature parameter controls how randomly the program selects the next note from the output of the softmax function.
![GUI](https://github.com/cwhite098/Midi_Composer_TF/blob/main/gui.png)

## Improvements
I hope to add, in the future, support for generating multi-instrument tracks and support for varying note velocity.
