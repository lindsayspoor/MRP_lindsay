from __future__ import division
from typing import Any
import numpy as np

import gymnasium as gym
#import gym
from gymnasium.utils import seeding
from gymnasium import spaces
import gymnasium
import sys, os
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import collections

from config import ErrorModel

### Environment
class ToricGameDynamicEnv(gym.Env):
    '''
    ToricGameEnv environment. Effective single player game.
    '''

    def __init__(self, settings):
        """
        Args:
            opponent: Fixed
            board_size: board_size of the board to use
        """

        self.settings=settings

        self.board_size = settings['board_size']
        self.error_model = settings['error_model']
        self.channels = [0]
        self.memory = False
        self.error_rate = settings['error_rate']
        self.mask_actions=settings['mask_actions']
        self.N = settings['N']
        self.iteration_step = settings['iteration_step']
        self.new_N=settings['new_N']

        # Keep track of the moves
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]

        self.counter=0 #counts the amount of steps taken by the agent
        # Empty State
        self.state = Board(self.board_size)
        self.done = None

        self.observation_space = spaces.MultiBinary(self.board_size*self.board_size) #3x3 plaquettes on which we can view syndromes
        self.action_space = spaces.discrete.Discrete(len(self.state.qubit_pos)+1) #last action = 'do nothing'


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]


    def action_masks(self):

        self.mask_qubits=[]
        self.action_masks_list=np.zeros((len(self.state.qubit_pos)+1))
        self.action_masks_list[:]=False

        if self.state.do_nothing()==True: #if there are no syndrome points on the board, the agent can only "do nothing"
            qubit_number = len(self.state.qubit_pos)
            self.action_masks_list[qubit_number]=True

        else:
            for i in self.state.syndrome_pos:
                a,b = i[0],i[1]
                mask_coords = [[(a-1)%(2*self.state.size),b%(2*self.state.size)],[a%(2*self.state.size),(b-1)%(2*self.state.size)],[a%(2*self.state.size),(b+1)%(2*self.state.size)],[(a+1)%(2*self.state.size),b%(2*self.state.size)]]
                for j in mask_coords:
                    qubit_number = self.state.qubit_pos.index(j)
                    self.mask_qubits.append(qubit_number)
                    self.action_masks_list[qubit_number]=True
                



        self.action_masks_list=list(self.action_masks_list)


        return self.action_masks_list





    def generate_errors(self):
        # Reset the board state
        self.state.reset()

        # Let the opponent do it's initial evil
        self.qubits_flips = [[],[]]
        self.initial_qubits_flips = [[],[]]
        self._set_initial_errors(self.N)
        self.action_masks_list=self.action_masks()

        self.done = self.state.has_logical_error(self.initial_qubits_flips)
        self.reward=1


        return self.state.encode(self.channels, self.memory)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
         super().reset(seed=seed, options=options)

         self.counter=0 #sets the amount of steps back to 0 when a new game starts
         initial_observation = self.generate_errors()
         return initial_observation, {'state': self.state, 'message':"reset"}



    def generate_new_errors(self):


        self._set_initial_errors(self.new_N)

        self.action_masks_list=self.action_masks()

        self.done = self.state.has_logical_error(self.initial_qubits_flips)


        return self.state.encode(self.channels, self.memory)



    def close(self):
        self.state = None

    def render(self, mode="human", close=False):
        fig, ax = plt.subplots()
        a=1/(2*self.board_size)

        for i, p in enumerate(self.state.plaquet_pos):
            if self.state.op_values[0][i]==1:
                fc='darkorange'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        for i, p in enumerate(self.state.star_pos):
            if self.state.op_values[1][i]==1:
                fc = 'green'
                plaq = plt.Polygon([[a*p[0], a*(p[1]-1)], [a*(p[0]+1), a*(p[1])], [a*p[0], a*(p[1]+1)], [a*(p[0]-1), a*p[1]] ], fc=fc)
                ax.add_patch(plaq)

        # Draw lattice
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos=(2*a*x, 2*a*y)
                width=a*2
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black' )
                ax.add_patch(lattice)

        for i, p in enumerate(self.state.qubit_pos):
            pos=(a*p[0], a*p[1])
            fc='darkgrey'
            if self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 0:
                fc='darkblue'
            elif self.state.qubit_values[0][i] == 0 and self.state.qubit_values[1][i] == 1:
                fc='darkred'
            elif self.state.qubit_values[0][i] == 1 and self.state.qubit_values[1][i] == 1:
                fc='darkmagenta'
            circle = plt.Circle( pos , radius=a*0.25, ec='k', fc=fc)
            ax.add_patch(circle)
            plt.annotate(str(i), pos, fontsize=8, ha="center")

        ax.set_xlim([-.1,1.1])
        ax.set_ylim([-.1,1.1])
        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

    def step(self, location,  without_illegal_actions=True):
        '''
        Args:
            location: coord of the qubit to flip
            operator: Pauli matrix to apply
        Return:
            observation: board encoding,
            reward: reward of the game,
            done: boolean,
            info: state dict
        '''

        if self.counter==300:
            print("max steps reached, terminated.")
            self.done=True
            return self.state.encode(self.channels, self.memory), 0, True, False,{'state': self.state, 'message':"max steps reached, terminated."}


        if (self.counter > 0) and (self.counter % self.iteration_step == 0):

            self.generate_new_errors()

        # If already terminal, then don't do anything, count as win
        if self.done:

            return self.state.encode(self.channels, self.memory), 0, True, False,{'state': self.state, 'message':"game over!"}

        # Check if we flipped twice the same qubit 
        pauli_opt=0
        pauli_X_flip=True
        pauli_Z_flip=False
        #self.render()

        self.counter+=1


        if  location == len(self.state.qubit_pos): #if the last action in the action space is chosen, the agent needs to do nothing in this case.
            if not self.state.has_logical_error(self.initial_qubits_flips):
                return self.state.encode(self.channels, self.memory), 1, False, False,{'state': self.state, 'message':"continue"}
            else:
                return self.state.encode(self.channels, self.memory), 0, True, False,{'state': self.state, 'message':"game over!"}
        

                

        else:
            if pauli_X_flip:
                self.qubits_flips[0].append(location) 

            if pauli_Z_flip:
                self.qubits_flips[1].append(location)

            self.state.act(self.state.qubit_pos[location], pauli_opt)

            if not self.state.has_logical_error(self.initial_qubits_flips):

                return self.state.encode(self.channels, self.memory), 1, False, False,{'state': self.state, 'message':"continue"}
            else:

                return self.state.encode(self.channels, self.memory), 0, True, False,{'state': self.state, 'message':"game over!"}
        


    def _set_initial_errors(self):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilitic mode
        # Pick random sites according to error rate
        for q in self.state.qubit_pos:    
            if np.random.rand() < self.error_rate:

                if self.error_model == ErrorModel["UNCORRELATED"]:
                    pauli_opt = 0
                elif self.error_model == ErrorModel["DEPOLARIZING"]:
                    pauli_opt = np.random.randint(0,3)

                pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
                pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


                if pauli_X_flip:
                    self.initial_qubits_flips[0].append( q )
                if pauli_Z_flip:
                    self.initial_qubits_flips[1].append( q )

                self.state.act(q, pauli_opt)


        # Now unflip the qubits, they're a secret
        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))

class ToricGameDynamicEnvFixedErrs(ToricGameDynamicEnv):
    def __init__(self, settings):
        super().__init__(settings)
        

    def _set_initial_errors(self, N):
        ''' Set random initial errors with an %error_rate rate
            but report only the syndrome
        '''
        # Probabilistic mode
        '''

        print(f"{self.state.qubit_values=}")
        q_options = []

        k=0
        for i in range(50):
            print("new error selection")
            if k==N:
                break

            qubit = np.random.choice(len(self.state.qubit_pos),1, replace=False)[0]
            qubit = self.state.qubit_pos[qubit]
            print(f"{qubit=}")
            print(f"{self.initial_qubits_flips[0]=}")
            print(f"{self.qubits_flips[0]=}")
            if qubit in self.initial_qubits_flips[0]: #dit klopt niet, dit moeten de huidige qubits op het board zijn die geflipt zijn
                print("continue qubit selection")
                continue
            else:
                q_options.append(qubit)
            k+=1

        '''
        for q in np.random.choice(len(self.state.qubit_pos), N, replace=False): 
        #for q in q_options:

            q = self.state.qubit_pos[q]

            if self.error_model == ErrorModel["UNCORRELATED"]:
                pauli_opt = 0
            elif self.error_model == ErrorModel["DEPOLARIZING"]:
                pauli_opt = np.random.randint(0,3)

            pauli_X_flip = (pauli_opt==0 or pauli_opt==2)
            pauli_Z_flip = (pauli_opt==1 or pauli_opt==2)


            if pauli_X_flip:
                self.initial_qubits_flips[0].append( q )
            if pauli_Z_flip:
                self.initial_qubits_flips[1].append( q )

            self.state.act(q, pauli_opt)
            #break
        # Now unflip the qubits, they're a secret

        self.state.qubit_values = np.zeros((2, 2*self.board_size*self.board_size))





class Board(object):
    '''
    Basic Implementation of a ToricGame Board, actions are int [0,2*board_size**2)
    o : qubit
    P : plaquette operator
    x : star operator

    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |
    x--o---x---o---x---o---
    |      |       |
    o  P   o   P   o   P
    |      |       |

    '''
    @staticmethod
    def component_positions(size):
        qubit_pos   = [[x,y] for x in range(2*size) for y in range((x+1)%2, 2*size, 2)]
        plaquet_pos = [[x,y] for x in range(1,2*size,2) for y in range(1,2*size,2)]
        star_pos    = [[x,y] for x in range(0,2*size,2) for y in range(0,2*size,2)]

        return qubit_pos, plaquet_pos, star_pos

    def __init__(self, board_size):
        self.size = board_size

        # Real-space locations
        self.qubit_pos, self.plaquet_pos, self.star_pos  = self.component_positions(self.size)

        # Mapping between 1-index and 2D position
        '''
        self.qubit_dict, self.star_dict, self.plaquet_dict = {},{},{}
        for i in range(2*self.size*self.size):
            self.qubit_dict[self.qubit_pos[i]] = i
        for i in range(self.size*self.size):
            self.star_dict[i] = self.star_pos[i]
            self.plaquet_dict[i] = self.plaquet_pos[i]
        '''

        # Define here the logical error for efficiency
        self.z1pos = [[0,x] for x in range(1, 2*self.size, 2)]
        self.z2pos = [[y,0] for y in range(1, 2*self.size, 2)]
        self.x1pos = [[1,x] for x in range(0, 2*self.size, 2)]
        self.x2pos = [[y,1] for y in range(0, 2*self.size, 2)]

        self.reset()

    def reset(self):
        #self.board_state = np.zeros( (2, 2*self.size, 2*self.size) )
        
        self.qubit_values = np.zeros((2, 2*self.size*self.size))
        self.op_values = np.zeros((2, self.size*self.size))

        self.syndrome_pos = [] # Location of syndromes

    def act(self, coord, operator):
        '''
            Args: input action in the form of position [x,y]
            coord: real-space location of the qubit to flip
        '''

        pauli_X_flip = (operator==0 or operator==2)
        pauli_Z_flip = (operator==1 or operator==2)



        #qubit_index = self.qubit_pos.index(coord)
        qubit_index=self.qubit_pos.index(coord)
        #coord = self.qubit_pos[coord]

        # Flip it!
        if pauli_X_flip:
            self.qubit_values[0][qubit_index] = (self.qubit_values[0][qubit_index] + 1) % 2
        if pauli_Z_flip:
            self.qubit_values[1][qubit_index] = (self.qubit_values[1][qubit_index] + 1) % 2

        # Update the syndrome measurements
        # Only need to incrementally change
        # Find plaquettes that the flipped qubit is a part of
        plaqs=[]
        if pauli_X_flip:
            if coord[0] % 2 == 0:
                plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]
            else:
                plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]

        if pauli_Z_flip:
            if coord[0] % 2 == 0:
                plaqs += [ [ coord[0], (coord[1] + 1) % (2*self.size) ], [ coord[0], (coord[1] - 1) % (2*self.size) ] ]
            else:
                plaqs += [ [ (coord[0] + 1) % (2*self.size), coord[1] ], [ (coord[0] - 1) % (2*self.size), coord[1] ] ]


        # Update syndrome positions
        for plaq in plaqs:
            if plaq in self.syndrome_pos:
                self.syndrome_pos.remove(plaq)
            else:
                self.syndrome_pos.append(plaq)

            # The plaquette or vertex operators are only 0 or 1
            if plaq in self.star_pos:
                op_index = self.star_pos.index(plaq)
                channel = 1
            elif plaq in self.plaquet_pos:
                op_index = self.plaquet_pos.index(plaq)
                channel = 0

        
            self.op_values[channel][op_index] = (self.op_values[channel][op_index] + 1) % 2





    def do_nothing(self):

        # Are all syndromes removed?
        return len(self.syndrome_pos) == 0 #if True the agent should do nothing

    def has_logical_error(self, initialmoves, debug=False):
        if debug:
            print("Initial errors:", [self.qubit_pos.index(q) for q in initialmoves])



        #check whether initial moves occur an even amount of times -> these should be removed as these result in an un-flipped qubit!

        for i in initialmoves[0]:
            counter = initialmoves[0].count(i)
            if (counter%2==0):
                initialmoves[0] = [j for j in initialmoves[0] if j != i]


        # Check for Z logical error
        zerrors = [0,0]
        for pos in self.z1pos:
            if pos in initialmoves[0]:
                zerrors[0] += 1
            qubit_index = self.qubit_pos.index(pos)
            zerrors[0] += self.qubit_values[0][ qubit_index ]

        for pos in self.z2pos:
            if pos in initialmoves[0]:
                zerrors[1] += 1
            qubit_index = self.qubit_pos.index(pos)
            zerrors[1] += self.qubit_values[0][ qubit_index ]

        # Check for X logical error
        xerrors = [0,0]
        for pos in self.x1pos:
            if pos in initialmoves[1]:
                xerrors[0] += 1
            qubit_index = self.qubit_pos.index(pos)
            xerrors[0] += self.qubit_values[1][ qubit_index ]

        for pos in self.x2pos:
            if pos in initialmoves[1]:
                xerrors[1] += 1
            qubit_index = self.qubit_pos.index(pos)
            xerrors[1] += self.qubit_values[1][ qubit_index ]

        #print("Zerrors", zerrors)


        if (zerrors[0]%2 == 1) or (zerrors[1]%2 == 1) or \
            (xerrors[0]%2 == 1) or (xerrors[1]%2 == 1):
            return True

        return False


    def __repr__(self):
        ''' representation of the board class
            print out board_state
        '''
        return f"Qubit Values: {self.qubit_values}, Operator values: {self.op_values}"

    def encode(self, channels, memory):
        '''Return: np array
            np.array(board_size, board_size): state observation of the board
        '''
        # In case of uncorrelated noise for instance, we don't need information
        # about the star operators

        img=np.array([])
        for channel in channels:
            img = np.concatenate((img, self.op_values[channel]))
            if memory:
                img = np.concatenate((img, self.qubit_values[channel]))

        return img

    def image_view(self, number=False, channel=0):
        image = np.empty((2*self.size, 2*self.size), dtype=object)
        #print(image)
        for i, plaq in enumerate(self.plaquet_pos):
            if self.op_values[0][i] == 1:
                image[plaq[0], plaq[1]] = "P"+str(i) if number else "P"
            elif self.op_values[0][i] == 0:
                image[plaq[0], plaq[1]] = "x"+str(i) if number else "x"
        for i,plaq in enumerate(self.star_pos):
            if self.op_values[1][i] == 1:
                image[plaq[0], plaq[1]] = "S"+str(i) if number else "S"
            elif self.op_values[1][i] == 0:
                image[plaq[0], plaq[1]] = "+"+str(i) if number else "+"

        for i,pos in enumerate(self.qubit_pos):
            image[pos[0], pos[1]] = str(int(self.qubit_values[channel,i]))+str(i) if number else str(int(self.qubit_values[channel, i]))

        return np.array(image)



