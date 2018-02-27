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
        self.transmission_prob = transmission_prob #transmission probabilities.

        self.emission_prob = emission_prob #emission probabilities.

        self.numberOfstate = transmission_prob.shape[1]
        #number of states are equal the different states for X, is equal the number of columbs of the transmission matrix.
        
        #an np.matrix object
        self.states_f = initial_states #this is our f value in the algorithm, set as f_0:0 to begin with. Should be transposed.
        # states[0] is the probability of it being rain, and states[1] is the probability of it being no rain.
        # Hold the previous step under calculation.
        
        self.b_messages = self.init_b_messages()  # an vector where every element is one. Hold the previous step under calculation.

        if initial_states == None:
            #If we don't have any intial states, we have to initialize to an normalised array of equal probabilities.
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
        onesM = np.ones(n_obsStates) #create an all ones matrix from this number
        return np.matrix([onesM*(1/n_obsStates)]).getT() #normalize and return.

    def init_b_messages(self):
        """ create an nx1 matrix with only 1s """
        return np.matrix(np.ones(self.numberOfstate)).getT()
    
    def observMatrix(self, observation):
        """ Create the observation matrix dependant on what observations we have observed (Observation model as matrix),
        since from the observations matrix, the diagonal corresponds to the observation of one event, columb 0 is umbrella, and columb 1 is no umbrella. """

        columb_eventProb = self.emission_prob[:,observation] #return an array of elemenst we are interested in.
        #print(columb_eventProb)
        #print(columb_eventProb)
        return np.matrix(np.diag(np.ravel(columb_eventProb))) #create an observation matrix of a given observation with the event matrix.
        """
            if observation is 0, umbrella: the observation matrix should look like [[0.9, 0],[0, 0.2]]
            if observation is 1, no umbrella: the obs_matrix should look like [[0.1 ,0],[0, 0.8]]
        """    

    def forward(self, observations):
        """ forward operation, calculating the forward messanges for given set of observations """
        forward_states = [] #forward messages
        forward_states.append(np.ravel(self.states_f.getT()).tolist()) #simply hold the intial state.(used in forward-backward)

        for observation in np.squeeze(observations.tolist()): #itterate tru every observation. 
            # f_t = O_t*T.transposed()(f_t-1).transpoed() 
            state_t = (self.observMatrix(observation).getT()*self.transmission_prob.getT()) * self.states_f #the calculation
            self.states_f = self.normalizer(state_t) #normalize the array
            #print("we got f_t: {}".format(self.states))
            forward_states.append(np.ravel(self.states_f.transpose()).tolist()) #store the normalized state
        self.states_f = np.matrix(forward_states[0]).getT() #set the initial states back
        return forward_states #Return list of the forward states.
        #state vector f_0:t = normalizer( f_0:t-1 T O_t ), which we can easily computet, since all 3 are matrixes
        #Itterate trou every obeservation, calculating the probable state at given time.
    
    def forward_i(self, f_prev , observation):#one itteration of forward algorithm, to match forward-backward algoritm.
        """ One itteration of the forward algoritm, but not used, since it works just as well doing them all at once. """
        forward_state = self.normalizer((self.observMatrix(observation)*self.transmission_prob.getT())*f_prev)
        return np.ravel(forward_state).tolist() #turn matrix into list.

    def backward_i(self,b_prev, observation):
        b_today = (self.transmission_prob*self.observMatrix(observation)*b_prev)
        return self.normalizer(b_today)

    def backward(self, observations):
        """ backwards algorithm. """
        backward_states = [] #backward messages.
        backward_states.append(np.ravel(self.b_messages).tolist()) #add our initial state of all ones.
        for i, observation in reversed(list(enumerate(np.ravel(observations).tolist()))): #itterate tru observations in reverse.
            state_t = (self.transmission_prob*self.observMatrix(observation)*self.b_messages) #perform the calculation
            self.b_messages = self.normalizer(state_t) # normalise the result
            backward_states.append(np.ravel(self.b_messages).tolist()) #add to backward messages
        self.b_messages = self.init_b_messages()
        backward_states.reverse() #need to reverse the list, bc index 0 is actually the last element at this stage, for later use.
        return backward_states
    
    def forward_backward(self, observations):
        """ Forward-Backward algorithm, calculates based on an given observation """

        s_vector = [] #smooted result
        #first element is f_0.
        f_messages = self.forward(observations) #f_messages from the forward algorithm, from k=0 to t. 
        #last element is initial b value.
        b_messages = self.backward(observations) #b_messages from the backward algorithm., frok k=0 to t.
        n_obs = observations.size
        for i in range(n_obs+1):
            #Basicly multply corresponding element in f_messages with the same i b_messanges, 
            # Since they are both represeted as an array, we have to transform to an np.matrix to run an multiplication.
            # This is basicly what we have to do for the matrixes to multiply together to [x_i*y_i, x_j,y_j, etc]
            # We run nx1 * nxm multiplication and pick out the diagonal to an 1xn matrix. 
            #sv =  np.matrix( np.diag( np.matrix(f_messages[i]).getT()*np.matrix(b_messages[i]) ) ).getT() #Old method.
            sv = np.multiply(np.matrix(f_messages[i]).getT(), np.matrix(b_messages[i]).getT())
            #Add the new result to s_vector.
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

emissions = np.matrix([[0.9, 0.1],[0.2, 0.8]]) #Emissions probabilites

model = HMM(transmissions, emissions) #create HMM model instance. We don't know the initial states, so we don't pass this one. 
#Part B 1
print("########## PART B 1 ############")
print("Task B_1 forward states normalized:t0 to t \n{} ".format(model.forward(observations_B_1)))
print("\n########## PART B 2 ############")
#Part B 2
observations_B_2 = np.matrix([0,0,1,0,0]) # umbrella, umbrella, no umbrella, umbrella, umbrella.
print("\Task B_2 forward states normalized from stat t0 to t \n{}".format(model.forward(observations_B_2)))

#Part C 1
observations_C_1 = observations_B_1 # umbrella, umbrella.
print("\n########## PART C 1 ############")
print("Task C_1 P(X_1|e_1:2) {}".format(model.forward_backward(observations_C_1)[1]) )
print("Task C_1 SV {}".format(model.forward_backward(observations_C_1)) )
print("Task C_1 backward messages: \n {}".format(model.backward(observations_C_1)))
print("Task C_1 forward messages: \n {}".format(model.forward(observations_C_1)))
#Part C 2
observations_C_2 = observations_B_2 # umbrella, umbrella, no umbrella, umbrella, umbrella.
print("\n########## PART C 2 ############")
print("Task C_2 backward messages: \n{}".format(model.backward(observations_C_2)) )
print("Task C_2 smoothed probability values: \n{}".format(model.forward_backward(observations_C_2)))
print("Task C_2 forward messages: \n{}".format(model.forward(observations_C_2)))