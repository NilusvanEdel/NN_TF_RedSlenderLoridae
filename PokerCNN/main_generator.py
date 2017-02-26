from game import create_bachtes
import sys
import os

#testing whether the whole process is speed up if the call is outside of the class
states_per_iteration =200
num_players = 2
create_bachtes(states_per_iteration, num_players)
print(states_per_iteration, " states created")
# if run with > 200 iterations it slows down, mem leak in pypokerengine?
python = sys.executable
os.execl(python, python, *sys.argv)
