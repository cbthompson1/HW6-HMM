import pytest
from hmm import HiddenMarkovModel
import numpy as np


def test_mini_weather():
    """
    Create an instance of your HMM class using the "small_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "small_weather_input_output.npz" file.

    Ensure that the output of your Forward algorithm is correct. 

    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    In addition, check for at least 2 edge cases using this toy model. 
    """

    mini_hmm = np.load('./data/mini_weather_hmm.npz')
    mini_input = np.load('./data/mini_weather_sequences.npz')

    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    expected_viterbi_sequence = mini_input['best_hidden_state_sequence']

    hmm = HiddenMarkovModel(
        observation_states,
        hidden_states,
        prior_p,
        transition_p,
        emission_p
    )

    forward = hmm.forward(mini_input['observation_state_sequence'])
    viterbi = hmm.viterbi(mini_input['observation_state_sequence'])

    assert round(forward, 5) == 0.03506
    assert np.array_equal(expected_viterbi_sequence, viterbi)

    # Toy case #1: No inputs
    forward = hmm.forward([])
    assert forward == 1
    viterbi = hmm.viterbi([])
    assert np.array_equal(viterbi, np.array([]))

    # Toy case #2: Bad inputs go into the creation of hmm
    bad_emission = emission_p
    bad_emission[0][0] = 0.0
    with pytest.raises(ValueError):
        hmm = HiddenMarkovModel(
            observation_states,
            hidden_states,
            prior_p,
            transition_p,
            emission_p
        )



def test_full_weather():

    """
    Create an instance of your HMM class using the "full_weather_hmm.npz" file. 
    Run the Forward and Viterbi algorithms on the observation sequence in the "full_weather_input_output.npz" file
        
    Ensure that the output of your Viterbi algorithm correct. 
    Assert that the state sequence returned is in the right order, has the right number of states, etc. 

    """

    mini_hmm = np.load('./data/full_weather_hmm.npz')
    mini_input = np.load('./data/full_weather_sequences.npz')

    hidden_states = mini_hmm['hidden_states']
    observation_states = mini_hmm['observation_states']
    prior_p = mini_hmm['prior_p']
    transition_p = mini_hmm['transition_p']
    emission_p = mini_hmm['emission_p']

    expected_viterbi_sequence = mini_input['best_hidden_state_sequence']

    hmm = HiddenMarkovModel(
        observation_states,
        hidden_states,
        prior_p,
        transition_p,
        emission_p
    )

    forward = hmm.forward(mini_input['observation_state_sequence'])
    viterbi = hmm.viterbi(mini_input['observation_state_sequence'])

    assert round(forward, 15) == 1.6865e-11
    assert np.array_equal(expected_viterbi_sequence, viterbi)













