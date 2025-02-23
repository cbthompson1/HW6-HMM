import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """

        # Ensure correct types.
        if type(observation_states) != np.ndarray:
            state = type(observation_states)
            raise TypeError(f"observation_states must be np array, but got '{state}'")
        if type(prior_p) != np.ndarray:
            state = type(prior_p)
            raise TypeError(f"prior_p must be np array, but got '{state}'")
        if type(hidden_states) != np.ndarray:
            state = type(hidden_states)
            raise TypeError(f"hidden_states must be np array, but got '{state}'")
        if type(transition_p) != np.ndarray:
            state = type(transition_p)
            raise TypeError(f"transition_p must be np array, but got '{state}'")
        if type(emission_p) != np.ndarray:
            state = type(emission_p)
            raise TypeError(f"emission_p must be np array, but got '{state}'")
        
        # validation of probabilities.
        if round(np.sum(prior_p), 5) != 1:
            raise ValueError("prior_p values do not sum to 1!")
        for row in transition_p:
            if round(np.sum(row), 5) != 1:
                raise ValueError("Not all rows in transition_p sum to 1")
        for row in emission_p:
            if round(np.sum(row), 5) != 1:
                raise ValueError("Not all rows in emission_p sum to 1")

        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p


    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        

        # Base case: If no observed status, p=1 since there is only one option.
        if len(input_observation_states) == 0:
            return 1

        # Step 1. Initialize variables - DP table with dimensions n & t.
        t = len(input_observation_states)
        n = len(self.hidden_states)
        alphas = np.zeros((t,n))

        # Use prior_p values for first states since there's no ability to go backwards at i=0
        for i in self.hidden_states_dict:
            emit_prob = self.emission_p[i, self.observation_states_dict[input_observation_states[0]]]
            alphas[0,i] = self.prior_p[i] * emit_prob
       
        # Step 2. Calculate probabilities - iterate through each state (t) and state option (n).
        # Of course, we already did i=0, so start at 1.
        for i in range(1, t):
            for j in range(n):
                emit_prob = self.emission_p[j, self.observation_states_dict[input_observation_states[i]]]
                alphas[i,j] = emit_prob * np.sum(alphas[i-1] * self.transition_p[:,j])

        # Step 3. Return final probability 
        return np.sum(alphas[t-1])


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states
        """

        # Base Case: return empty array if no observations.
        if len(decode_observation_states) == 0:
            return np.array([])
        
        # Step 1. Initialize variables
        states = self.hidden_states_dict
        priors = self.prior_p
        trans = self.transition_p
        emits = self.emission_p
        obs = self.observation_states_dict

        t = len(decode_observation_states)
        n = len(self.hidden_states)
       
        # Step 2. Calculate Probabilities
        
        # Store probabilities of hidden state at each step 
        viterbi_table = np.zeros((t, n))
        prev = np.zeros((t, n), dtype=int)
        
        # i=0 case using priors input:
        for state in states:
            viterbi_table[0, state] = priors[state] * emits[state, obs[decode_observation_states[0]]]

        # general i=1 --> i=T case:
        for i in range(1, t):
            for s1 in states: # curr state
                for s2 in states: # prev. state
                    # Calculate the probability of transitioning from s2 to s1 and emitting the observation.
                    new_prob = viterbi_table[i-1,s2] * trans[s2,s1] * emits[s1,obs[decode_observation_states[i]]]
                    if new_prob > viterbi_table[i,s1]:
                        viterbi_table[i,s1] = new_prob
                        prev[i,s1] = s2
            
        # Step 3. Traceback

        # store best path for traceback
        path = np.zeros(t, dtype=int)
        # Start from the hidden state at the last iteration with the highest probability.
        path[-1] = np.argmax(viterbi_table[-1])
        # Use backpointing to get previous state.
        for t in range(0, t-1)[::-1]:
            path[t] = prev[t+1,path[t+1]]

        # Step 4. Return best hidden state sequence
        return np.array([self.hidden_states_dict[state] for state in path])
        