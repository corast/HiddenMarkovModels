
import numpy as np
#Need numpy for matrix calculations
class HMM():

    def __init__(self, transmission_prob, emission_prob, initial_states = None):
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
        self.transmission_prob = transmission_prob

        self.emission_prob = emission_prob

        self.numberOfstate = transmission_prob.shape[1]
        #number of states are equal the different states for X, is equal the number of columbs of the transmission matrix.

        self.states = initial_states #this is our f value in the algorithm, set as f_0:0 to begin with. Should be transposed.
        # states[0] is the probability of it being rain, and states[1] is the probability of it being no rain.

        self.forward_states = [] #An matrix that will simply hold the probability of each state being present at time t. e.g f_0:1 is at time 1. f_0:t is at state t.

        self.observations = observations #Holds the observations at the different times t.

        if initial_states == None:
            #print("initial states are None")
            n_obsStates = emission_prob.shape[1] #collect number of columbs in the emission matrix, there should be one per observable event.
            onesM = np.ones(n_obsStates)
            states = np.matrix([onesM*(1/n_obsStates)])
            self.states = np.matrix([onesM*(1/n_obsStates)]).getT()
        else:
            #Check if the inital states are f_normalized
            if(initial_states.shape[1] != 1):
                self.states = self.states.transpose()
            f_t = self.states.transpose()

            if(f_t*self.states != 1):
                print("Error, initial states are not normalized, all values should be equal to 1")
                return


    def init_obs_states(self):
        """ Initiate the initial states, if we don't know these, we set them as being equally probable """
        n_obsStates = self.emission_prob.shape[1]
        onesM = np.ones(n_obsStates)
        return onesM*(1/n_obsStates).transpose()
    
    def observMatrix(self, observation):
        """ Create the observation matrix dependant on what observations we have observed (Observation model as matrix),
        since from the observations matrix, the diagonal corresponds to the observation of one event, columb 0 is umbrella, and columb 1 is no umbrella. """

        columb_eventProb = self.emission_prob[:,observation]
        #print(columb_eventProb)
        return np.diag(np.squeeze(columb_eventProb.tolist())) #create an observation matrix of a given observation with the event matrix.
        """
            if observation is 0: the observation matrix should look like [[0.9, 0][0, 0.2]]
            if observation is 1: the obs_matrix should look like [[0.1 ,0],[]]
        """

    def forward(self, observations):
        """ forward operation """
        self.forward_states.append(np.ravel(self.states.transpose()).tolist()) #simply hold the intial state.

        for observation in np.squeeze(observations.tolist()): #itterate tru every observation. 
            #print("O_t * T^t * f_t^t: \n{} {} {} ".format(self.observMatrix(observation), self.transmission_prob.transpose(), self.states))
            state_t = (self.observMatrix(observation).transpose()*self.transmission_prob.getT()) * self.states
            self.states = self.normalizer(state_t) #normalize the array
            #print("we got f_t: {}".format(self.states))
            self.forward_states.append(np.ravel(self.states.transpose()).tolist()) #store the normalized state

        #state vector f_0:t = normalizer( f_0:t-1 T O_t ), which we can easily computet, since all 3 are matrixes
        #Itterate trou every obeservation, calculating the probable state at given time.

    def normalizer(self, f):
        """ Takes a state f transposed, and normalizes it so that the sum of all possible states are 1.
        this is done by taking 1 and divide by the sum of the unnormalized states f.
        e.g. say we have f = [0.5645 0.0745], the sum of f is 0.639,
            1/0.639 = 1.5649.
            f_normalized =  1.5649*[0.5645 0.0745] = 0.8834 0.1166
        """
        c = 1/f.sum()
        return c*f #normalize by multiplying with normalizing constant

transmissions = np.matrix([[0.7, 0.3],[0.3, 0.7]]) #equal to the dynamic model

observations = np.matrix([[0,0]]) #0 is equals umbrella, 1 is equal no umbrella

emissions = np.matrix([[0.9, 0.1],[0.2, 0.8]])

#print(emissions.tolist())
#We don't know the initial states, so we don't pass this one.

#np.array[row][columb]

#Testings

model = HMM(transmissions, emissions)
model.forward(observations)

model_2 = HMM(transmissions,emissions)
model_2.forward(np.matrix([0,0,1,0,0]))
print(model_2.forward_states)

"""
obs_2 = np.matrix([[1,2]])
obs = np.array([[1, 2]])
print(obs)
print(np.squeeze(obs).tolist())
print(obs_2)
print(np.squeeze(obs_2.tolist()).tolist())
print(np.ravel(obs_2).tolist())

print("\n")

ma_test = np.matrix([0.9, 0.2])
print(ma_test.tolist())
print(np.squeeze(ma_test.tolist()))
print(np.diag(np.squeeze(ma_test.tolist())))

print("\n")
f_0 = np.matrix([0.5, 0.5]).transpose()
result_1 = np.diag(np.squeeze(ma_test.tolist()))*transmissions.getT()*f_0
print(result_1)
"""