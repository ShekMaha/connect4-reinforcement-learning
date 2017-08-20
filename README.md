# Connect 4 with Reinforcement Learning

This is a repository for training an AI agent to play Connect 4 using reinforcement learning. Currently, a Q-learning RL algorithm. This also supports 2 player mode.

## Dependencies

Pygame for python2.7

## Running the Program

python2.7 connect4.py [iterations] 

(iterations is the number of iterations you want to train the computer which plays against itself) By default it is 20 if an argument is not provided

To train the agent, select Train Computer in the main menu. It will play iterations games which was passed as an argument to the program. After training the computer, when 'vs Computer' option is selected, a human can play against the trained computer. Note that each time 'Train Computer' mode is selected, it trains from the beginning.

## Controls

Simply use left/right arrow keys to navigate across different columns of the board and press enter to drop a coin.
