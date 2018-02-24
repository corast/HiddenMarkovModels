
import numpy as np
#Need numpy for matrix calculations
class HMM():

    def __init__(self,transmission, emission, init_obs):
        """
        transmission is the probabilites P(X_t|X_[t-1]) of a given random variable X_t representing all possible states in the HMM
        , i.e. the probabilites of changing state from one state to the others.
        An T x T matrix, where T is the number of states possible.

        emission is the probabilites assoiated with P(E | X_t) of what is the probability of observing E, given the state at t.
        , i.e. the probabilites of making an observation E given a certain state. If its raining, what is the probability we observe an umbrella or no umbrella.
        An M x T matrix, where T is the number of possible states, and M is the possible number of observations. 
        In our case we have 2 states, raining or not raining, and 2 possible observations, umbrella or no umbrella.

        init_obs is the initial observation probabilites, if we don't know each possible observation we set it to 0.5s for each observation element.
        an 1 x M matris, where M is all possible observation labels. 
        """

        self.obs = init_obs;
        self.forward_states = [] #An matrix that will simply hold the probability of each state being present at time t. e.g f_0:1 is at time 1. f_0:t is at state t.

    def forward(self):
        """ forward operation """

        #state vector f_0:t = normalizer( f_0:t-1 T O_t ), which we can easily computet, since all 3 are matrixes


        pass

    def normalizer(self):
        """ Takes a state f, and normalizes it so that the sum of all possible states are 1.
        this is done by taking 1 and divide by the sum of the unnormalized states f.
        e.g. say we have f = [0.5645 0.0745], the sum of f is 0.639, can be done by doing an matrix multiplication of an 1xM matrix
            f: 1xm * mx1 = 1x1 matrix, an scalar holding the sum.

            1/0.639 = 1.5649.
            f_normalized =  1.5649*[0.5645 0.0745] = 0.8834 0.1166
        """

        pass


transmissions = np.array([[0.7, 0.3],[0.3,0.7]])
observations = np.array(['1','2']) #We have two observations, umbrella or no umbrella
#np.array[row][columb]
print(observations[0])
