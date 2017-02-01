## Synopsis

This is the GitHub Page for one of the groups of the course Implementing Neural Networks with Tensorflow, the main project is a Convolutional Neural Network for Texas Hold'em Poker (9p, cash-game, no limit), which is currently in progress.


## Code Example


## Motivation

The PokerCNN should finally be able to learn 9 player cash-game no limited in Poker. The training should be provided by the creation of a poker engine implemented via the PyPokerEngine library (https://github.com/ishikota/PyPokerEngine), which will be played by heuristic bots. The handevaluation (no odd-calculation, the NN should be able to learn this by itself) provides the deuces library (https://github.com/worldveil/deuces)

## Installation

In order to use/ code on the CNN-PokerBot and the game engine it is necessary to install the PyPokerEngine library (pip install PyPokerEngine) and the deuces library (pip install deuces).
Afterwards the library has to be converted to Python3, this can either be done manually (2to3 convert) or the libraries can be replaced by the files provided (highly recommened due to a fix of the PokerEngine in regards to the minmal raise).


## Tests

Not working yet.
