from game import create_bachtes
import sys
import os

# to run this program the adapted libraries "PyPokerEngine" and "deuces" are required
# optimally the normal libraries should be installed (apt install) and be replaced by the provided files
#the main class to generate data
states_per_iteration =200
num_players = 2
create_bachtes(states_per_iteration, num_players)
print(states_per_iteration, " states created")
# if run with > 200 iterations the generation usually slows down, probably caused by memory leaks in the Pypokerengine
# this will restart the whole programm, notice there is no automatic shutdown possible
python = sys.executable
os.execl(python, python, *sys.argv)
