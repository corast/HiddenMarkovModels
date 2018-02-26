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
        
        #an np.matrix object
        self.states_f = initial_states #this is our f value in the algorithm, set as f_0:0 to begin with. Should be transposed.
        # states[0] is the probability of it being rain, and states[1] is the probability of it being no rain.
        
        self.b_messages = self.init_b_messages()  # an vector where every element is one.

        #simple list
        #self.forward_states = [] #An list that will simply hold the probability of each state being present at time t. e.g f_0:1 is at time 1. f_0:t is at state t.

        #self.backward_states = [] #An list that wil hold the each state from the backwards algorithm.

        if initial_states == None:
            self.states_f = self.init_obs_states()
        else:
            #Check if the inital states are f_normalized
            if(initial_states.shape[1] != 1):
                self.states_f = self.states_f.getT()
            f_t = self.states_f.getT()

            if(f_t*self.states_f != 1):
                print("Error, initial states are not normalized, all values should be equal to 1")
                return

    def init_obs_states(self):
        """ Initiate the initial states, if we don't know these, we set them as being equally probable """
        n_obsStates = self.emission_prob.shape[1] #collect number of columbs in the emission matrix, there should be one per observable event.
        onesM = np.ones(n_obsStates)
        return np.matrix([onesM*(1/n_obsStates)]).getT()

    def init_b_messages(self):
        return np.matrix(np.ones(self.numberOfstate)).getT()
    
    def observMatrix(self, observation):
        """ Create the observation matrix dependant on what observations we have observed (Observation model as matrix),
        since from the observations matrix, the diagonal corresponds to the observation of one event, columb 0 is umbrella, and columb 1 is no umbrella. """

        columb_eventProb = self.emission_prob[:,observation]
        #print(columb_eventProb)
        #print(columb_eventProb)
        return np.matrix(np.diag(np.ravel(columb_eventProb))) #create an observation matrix of a given observation with the event matrix.
        """
            if observation is 0: the observation matrix should look like [[0.9, 0][0, 0.2]]
            if observation is 1: the obs_matrix should look like [[0.1 ,0],[]]
        """    
    def obsMatrix(self, observation):
        columb_eventProb = self.emission_prob[:,observation]
        print(columb_eventProb)
        a = np.matrix(np.diag(np.ravel(columb_eventProb)))
        print(a)
        return a

    def forward(self, observations):
        """ forward operation """
        forward_states = []
        forward_states.append(np.ravel(self.states_f.getT()).tolist()) #simply hold the intial state.

        for observation in np.squeeze(observations.tolist()): #itterate tru every observation. 
            #print("O_t * T^t * f_t^t: \n{} {} {} ".format(self.observMatrix(observation), self.transmission_prob.transpose(), self.states))
            state_t = (self.observMatrix(observation).getT()*self.transmission_prob.getT()) * self.states_f
            self.states_f = self.normalizer(state_t) #normalize the array
            #print("we got f_t: {}".format(self.states))
            forward_states.append(np.ravel(self.states_f.transpose()).tolist()) #store the normalized state
        self.states_f = np.matrix(forward_states[0]).getT() #set the initial states back
        return forward_states #Return list of the forward states.
        #state vector f_0:t = normalizer( f_0:t-1 T O_t ), which we can easily computet, since all 3 are matrixes
        #Itterate trou every obeservation, calculating the probable state at given time.
    
    def forward_i(self, f_prev , observation):#one itteration of forward algorithm, to match forward-backward algoritm.
        print("input: {} {} ".format(f_prev, observation))
        print("shapes: {} {} {}".format(self.observMatrix(observation).shape,  self.transmission_prob.getT().shape, f_prev.shape))
   
        forward_state = self.normalizer((self.observMatrix(observation)*self.transmission_prob.getT())*f_prev)
        return np.ravel(forward_state).tolist() #turn matrix into list.

    def backward(self, observations):
        backward_states = []
        backward_states.append(np.ravel(self.b_messages).tolist())
        for i, observation in reversed(list(enumerate(np.ravel(observations).tolist()))):
            state_t = (self.transmission_prob*self.observMatrix(observation)*self.b_messages)
            self.b_messages = self.normalizer(state_t)
            backward_states.append(np.ravel(self.b_messages).tolist())
        self.b_messages = self.init_b_messages()
        backward_states.reverse() #need to reverse the list, bc index 0 is actually the last element at this stage.
        return backward_states
    
    def forward_backward(self, observations):

        #f_messages = []
        #f_messages.append(np.ravel(self.states_f).tolist())
        b_messages = self.b_messages
        s_vector = [] #smooted result
        #for observation in np.ravel(observations):
        #    #print("f_prev {} ".format(np.matrix(f_messages[-1]).getT()))
        #    f_messages.append(self.forward_i( np.matrix(f_messages[-1]).getT(), observation ))
        #return f_messages
        f_messages = self.forward(observations)
        b_messages = self.backward(observations)
        n_obs = observations.size
        for i in range(n_obs+1):
            
            #print("\ni:{} {} * {}\n ".format(i,np.matrix(f_messages[i]).getT(),np.matrix(b_messages[i]).getT()))
            sv =  np.matrix( np.diag( np.matrix(f_messages[i]).getT()*np.matrix(b_messages[i]) ) ).getT() 

            s_vector.append(np.ravel(self.normalizer(sv).getT()).tolist())
        return s_vector

    def normalizer(self, v):
        """ Takes a vector v transposed, and normalizes it so that the sum of all possible states are 1.
        this is done by taking 1 and divide by the sum of the unnormalized states f.
        e.g. say we have f = [[0.5645], [0.0745]], the sum of f is 0.639,
            1/0.639 = 1.5649.
            f_normalized =  1.5649*[[0.5645], [0.0745]] = 0.8834 0.1166
        """
        c = 1/v.sum()
        return c*v #normalize by multiplying with normalizing constant

#Part B
transmissions = np.matrix([[0.7, 0.3],[0.3, 0.7]]) #equal to the dynamic model

observations_B_1 = np.matrix([[0,0]]) #0 is equals umbrella, 1 is equal no umbrella

emissions = np.matrix([[0.9, 0.1],[0.2, 0.8]])

model = HMM(transmissions, emissions) #create HMM model instance. We don't know the initial states, so we don't pass this one. 

#print(model.forward(observations_B_1))
#print(model.forward_states)

observations_B_2 = np.matrix([0,0,1,0,0])
#print(model.forward(observations_B_2))

#Part C
observations_C_1 = observations_B_1
#print(model.backward(observations_C_1))

print(model.forward_backward(observations_C_1))
"""
a = np.matrix([[0.5], [0.5],[0.5]])
b = np.matrix([[0.6469],[0.3513],[0.2]])
c = a*b.getT()
obs_s = 3
ident = np.matrix(np.identity(3))
ind = np.ones(3)
print(ind*c)
print(np.ravel(c.diagonal().transpose()).tolist())
"""

#c = np.identity((a*b.getT()).size[0])
#print(c)
#print(b.shape)
#print(a*b.getT())
#print(np.matrix(np.diag(a*b.getT())).getT())

#c = a*b.getT()

#d = np.matrix([[1,2,3]])
#print(d)
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