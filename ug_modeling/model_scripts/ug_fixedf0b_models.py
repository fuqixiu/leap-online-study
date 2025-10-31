import numpy as np
from tqdm import tqdm
import sys
sys.path.append('C:/Users/fuq01/Documents/GitHub/pyEM')
from pyEM.math import softmax, norm2alpha
from scipy.special import expit

############################## Simulation function ######################################
def simulate_RW(params, offers, nblocks=1, ntrials=30, varbeta=False, varf0=False):
    """
    Simulate the basic RW model.
    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `offers` is a np.array of shape (nblocks, ntrials)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials, 2))
    vd          = np.zeros((nsubjects, nblocks, ntrials,))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    all_norms   = np.zeros((nsubjects, nblocks, ntrials+1,))
    all_offers  = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    f0 = 10 # fixed initial norm

    for subj_idx in tqdm(range(nsubjects)):
        envy, lr = params[subj_idx,:2]

        if varbeta:
            beta = params[subj_idx, 2]
        else:
            beta = 1 # fixed beta

        if varf0:
            f0 = params[subj_idx, 2]

        for b in range(nblocks): # if nblocks == 1, then use reversals
            np.random.seed(subj_idx * nblocks + b)
            shuffled_offers = np.random.permutation(offers)
            all_offers[subj_idx,b,:] = shuffled_offers.copy()

            norms = RW(f0, lr, all_offers[subj_idx,b,:])
            all_norms[subj_idx,b,:] = norms.copy()

            for t in range(ntrials):

                # compute EV
                ev[subj_idx, b, t, :] = [all_offers[subj_idx, b, t], envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0)] # accept, reject
                vd[subj_idx, b, t]  = all_offers[subj_idx, b, t] - envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0) # accept minus reject
                
                # normed version vd
                # vd[subj_idx, b, t]  = all_offers[subj_idx, b, t] - envy * np.max(all_norms[subj_idx, b, t+1]*20 - all_offers[subj_idx, b,t], 0) # accept minus reject

                # calculate choice probability
                # ch_prob[subj_idx, b, t, :] = softmax(ev[subj_idx, b, t, :], beta)
                prob_accept = 1 / ( 1 + np.exp(-beta * vd[subj_idx, b, t]))
                # prob_accept = 1 / ( 1 + np.exp(vd[subj_idx, b, t]/beta))
                ch_prob[subj_idx, b, t, 0] = prob_accept.copy()
                ch_prob[subj_idx, b, t, 1] = 1 - prob_accept.copy()

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['A', 'R'], 
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t, :])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A': # accept
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    rewards[subj_idx, b, t]   = all_offers[subj_idx,b,t]
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    rewards[subj_idx, b, t] = 0

                choice_nll[subj_idx, b, t] = -np.log(ch_prob[subj_idx, b, t, c].copy())

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'vd'        : vd, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'offers'    : all_offers,
                 'norms'     : all_norms,
                 'choice_nll': choice_nll}

    return subj_dict

def simulate_FS_fixb(params, offers, nblocks=1, ntrials=30):
    """
    Simulate the basic FS model.
    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `offers` is a np.array of shape (nblocks, ntrials)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials, 2))
    vd          = np.zeros((nsubjects, nblocks, ntrials,))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    all_norms   = np.zeros((nsubjects, nblocks, ntrials+1,))
    all_offers  = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    for subj_idx in tqdm(range(nsubjects)):
        envy, f0 = params[subj_idx,:2]
        beta = 1

        for b in range(nblocks):
            # generating offers
            np.random.seed(subj_idx * nblocks + b)
            shuffled_offers = np.random.permutation(offers)
            all_offers[subj_idx,b,:] = shuffled_offers.copy()

            # initial norm
            all_norms[subj_idx,b,:] = f0

            for t in range(ntrials):
                # compute EV
                ev[subj_idx, b, t, :] = [all_offers[subj_idx, b, t], envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0)] # accept, reject
                vd[subj_idx, b, t]    = all_offers[subj_idx, b, t] - envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0) # accept minus reject
                
                # calculate choice probability
                prob_accept = 1 / ( 1 + np.exp(-beta * vd[subj_idx, b, t]))
                ch_prob[subj_idx, b, t, 0] = prob_accept.copy()
                ch_prob[subj_idx, b, t, 1] = 1 - prob_accept.copy()

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['A', 'R'], 
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t, :])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A': # accept
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    rewards[subj_idx, b, t]   = all_offers[subj_idx,b,t]
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    rewards[subj_idx, b, t] = 0

                choice_nll[subj_idx, b, t] = -np.log(ch_prob[subj_idx, b, t, c].copy())

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'vd'        : vd, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'offers'    : all_offers,
                 'norms'     : all_norms,
                 'choice_nll': choice_nll}

    return subj_dict

def simulate_FS_fixf0(params, offers, nblocks=1, ntrials=30):
    """
    Simulate the basic FS model.
    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `offers` is a np.array of shape (nblocks, ntrials)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials, 2))
    vd          = np.zeros((nsubjects, nblocks, ntrials,))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    all_norms   = np.zeros((nsubjects, nblocks, ntrials+1,))
    all_offers  = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    for subj_idx in tqdm(range(nsubjects)):
        envy, beta = params[subj_idx,:2]
        f0 = 10 # fixed initial norm
        # beta = 1 # fixed beta

        for b in range(nblocks):
            # generating offers
            np.random.seed(subj_idx * nblocks + b)
            shuffled_offers = np.random.permutation(offers)
            all_offers[subj_idx,b,:] = shuffled_offers.copy()

            # initial norm
            all_norms[subj_idx,b,:] = f0

            for t in range(ntrials):
                # compute EV
                ev[subj_idx, b, t, :] = [all_offers[subj_idx, b, t], envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0)] # accept, reject
                vd[subj_idx, b, t]    = all_offers[subj_idx, b, t] - envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0) # accept minus reject
                
                # calculate choice probability
                prob_accept = 1 / ( 1 + np.exp(-beta * vd[subj_idx, b, t]))
                ch_prob[subj_idx, b, t, 0] = prob_accept.copy()
                ch_prob[subj_idx, b, t, 1] = 1 - prob_accept.copy()

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['A', 'R'], 
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t, :])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A': # accept
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    rewards[subj_idx, b, t]   = all_offers[subj_idx,b,t]
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    rewards[subj_idx, b, t] = 0

                choice_nll[subj_idx, b, t] = -np.log(ch_prob[subj_idx, b, t, c].copy())

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'vd'        : vd, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'offers'    : all_offers,
                 'norms'     : all_norms,
                 'choice_nll': choice_nll}

    return subj_dict

def simulate_FS(params, offers, nblocks=1, ntrials=30):
    """
    Simulate the basic FS model.
    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `offers` is a np.array of shape (nblocks, ntrials)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials, 2))
    vd          = np.zeros((nsubjects, nblocks, ntrials,))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    all_norms   = np.zeros((nsubjects, nblocks, ntrials+1,))
    all_offers  = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    for subj_idx in tqdm(range(nsubjects)):
        envy, beta, f0 = params[subj_idx,:3]

        for b in range(nblocks):
            # generating offers
            np.random.seed(subj_idx * nblocks + b)
            shuffled_offers = np.random.permutation(offers)
            all_offers[subj_idx,b,:] = shuffled_offers.copy()

            # initial norm
            all_norms[subj_idx,b,:] = f0

            for t in range(ntrials):
                # compute EV
                ev[subj_idx, b, t, :] = [all_offers[subj_idx, b, t], envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0)] # accept, reject
                vd[subj_idx, b, t]    = all_offers[subj_idx, b, t] - envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0) # accept minus reject
                
                # calculate choice probability
                prob_accept = 1 / ( 1 + np.exp(-beta * vd[subj_idx, b, t]))
                ch_prob[subj_idx, b, t, 0] = prob_accept.copy()
                ch_prob[subj_idx, b, t, 1] = 1 - prob_accept.copy()

                # make choice
                choices[subj_idx, b, t]   = np.random.choice(['A', 'R'], 
                                                size=1, 
                                                p=ch_prob[subj_idx, b, t, :])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A': # accept
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    rewards[subj_idx, b, t]   = all_offers[subj_idx,b,t]
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    rewards[subj_idx, b, t] = 0

                choice_nll[subj_idx, b, t] = -np.log(ch_prob[subj_idx, b, t, c].copy())

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'vd'        : vd, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'offers'    : all_offers,
                 'norms'     : all_norms,
                 'choice_nll': choice_nll}

    return subj_dict

def simulate_FS_noBeta(params, offers, nblocks=1, ntrials=30):
    """
    Simulate the basic FS model.
    Args:
        `params` is a np.array of shape (nsubjects, nparams)
        `offers` is a np.array of shape (nblocks, ntrials)
        `nblocks` is the number of blocks to simulate
        `ntrials` is the number of trials per block
    Returns:
        `simulated_dict` is a dictionary with the simulated data with the following keys:
            - `ev` is a np.array of shape (nsubjects, nblocks, ntrials+1, 2)
            - `ch_prob` is a np.array of shape (nsubjects, nblocks, ntrials, 2)
            - `choices` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choices_A` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `rewards` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `pe` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `choice_nll` is a np.array of shape (nsubjects, nblocks, ntrials)
            - `params` is a np.array of the parameters used to simulate the data
                - `beta` is the softmax inverse temperature
                - `lr` is the learning rate
    """
    nsubjects   = params.shape[0]
    ev          = np.zeros((nsubjects, nblocks, ntrials, 2))
    vd          = np.zeros((nsubjects, nblocks, ntrials,))
    ch_prob     = np.zeros((nsubjects, nblocks, ntrials,   2))
    choices     = np.empty((nsubjects, nblocks, ntrials,), dtype='object')
    choices_A   = np.zeros((nsubjects, nblocks, ntrials,))
    rewards     = np.zeros((nsubjects, nblocks, ntrials,))
    all_norms   = np.zeros((nsubjects, nblocks, ntrials+1,))
    all_offers  = np.zeros((nsubjects, nblocks, ntrials,))
    choice_nll  = np.zeros((nsubjects, nblocks, ntrials,))

    subj_dict = {}
    for subj_idx in tqdm(range(nsubjects)):
        envy, f0 = params[subj_idx,:2]

        for b in range(nblocks):
            # generating offers
            np.random.seed(subj_idx * nblocks + b)
            shuffled_offers = np.random.permutation(offers)
            all_offers[subj_idx,b,:] = shuffled_offers.copy()

            # initial norm
            all_norms[subj_idx,b,:] = f0

            for t in range(ntrials):
                # compute EV
                ev[subj_idx, b, t, :] = [all_offers[subj_idx, b, t], envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0)] # accept, reject
                vd[subj_idx, b, t]    = all_offers[subj_idx, b, t] - envy * np.max(all_norms[subj_idx, b, t+1] - all_offers[subj_idx, b,t], 0) # accept minus reject
                
                # calculate choice probability
                # prob_accept = 1 / ( 1 + np.exp(-beta * vd[subj_idx, b, t]))
                # ch_prob[subj_idx, b, t, 0] = prob_accept.copy()
                # ch_prob[subj_idx, b, t, 1] = 1 - prob_accept.copy()

                # make choice
                choices_idx = np.argmax(ev[subj_idx, b, t, :])
                choices[subj_idx, b, t]   = ['A', 'R'][choices_idx]
                
                # choices[subj_idx, b, t]   = np.random.choice(['A', 'R'], 
                #                                 size=1, 
                #                                 p=ch_prob[subj_idx, b, t, :])[0]

                # get choice index
                if choices[subj_idx, b, t] == 'A': # accept
                    c = 0
                    choices_A[subj_idx, b, t] = 1
                    rewards[subj_idx, b, t]   = all_offers[subj_idx,b,t]
                else:
                    c = 1
                    choices_A[subj_idx, b, t] = 0
                    rewards[subj_idx, b, t] = 0

                choice_nll[subj_idx, b, t] = -np.log(ch_prob[subj_idx, b, t, c].copy())

    # store params
    subj_dict = {'params'    : params,
                 'ev'        : ev, 
                 'vd'        : vd, 
                 'ch_prob'   : ch_prob, 
                 'choices'   : choices, 
                 'choices_A' : choices_A, 
                 'rewards'   : rewards, 
                 'offers'    : all_offers,
                 'norms'     : all_norms,
                 'choice_nll': choice_nll}

    return subj_dict

############################### Utility functions ######################################
def RW(f0, lr, offers, norm=False):
    # Norm adaptation - RW model, f is the norm update
    n_trials = len(offers)
    
    f = np.zeros(n_trials+1)
    # f[0] = f0

    if norm: # percentage split of the norm
        offers = offers.copy()/20
        f[0] = f0/20
    else:
        f[0] = f0 #default version

    for t in range(n_trials):
        f[t + 1] = f[t] + lr * (offers[t] - f[t])
    
    return f

def norm2f0(x):
    return 20 / (1 + np.exp(-x))

def norm2envy(x):
    return 10 / (1 + np.exp(-x))

def envy2norm(x):
    return np.log(x / (10 - x))

def norm2beta(x):
    return (1.1 / (1 + np.exp(-x)))

def beta2norm(x):
    return np.log(x / (1.1 - x))

############################### Fitting functions ######################################
def fit(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the basic RW model to a single subject's data. [template version]
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial (what is this about?)
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    # offers = offers.copy() / 10 #normalize offers 
    f0 = 10 #/ 10
    beta = 1
    nparams = len(params)   
    envy = norm2envy(params[0])
    lr   = norm2alpha(params[1])

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        print(f'lr = {lr:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        
        norms = RW(f0, lr, offers[b,:])
        all_norms[b,:] = norms.copy()

        for t in range(ntrials):

            # compute EV
            ev[b, t, 0] = offers[b,t] # accept (0)
            ev[b, t, 1] = envy * np.max(all_norms[b,t+1] - offers[b,t], 0) # reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A': #accept
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            # ch_prob[b, t,:] = softmax(ev[b, t, :], beta)
            # if `np.exp(-beta * vd[b, t])` is too large, then return 1; if too small, return 0; else compute the probability
            # if (-beta * vd[b, t]) > 5:
            #     prob_accept = np.array([0.0])
            # elif (-beta * vd[b, t]) < -5:
            #     prob_accept = np.array([1.0])
            # else:
            #     prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))

            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t])) #using value difference to calculate probability of accepting
            # prob_accept = 1 / ( 1 + np.exp(vd[b, t]/beta))
            ch_prob[b, t, 0] = prob_accept.copy() #accept
            ch_prob[b, t, 1] = 1 - prob_accept.copy() #reject
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_RW_varf0(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the RW model with variable initial norm and fix beta at 1 to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial (what is this about?)
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)   
    envy = norm2envy(params[0])
    lr   = norm2alpha(params[1])
    beta = 1
    f0   = norm2f0(params[2]) #double check

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        print(f'lr = {lr:.3f} not in range')
        return 10000000
    this_f0_bounds = [0.1, 20]
    if f0 < min(this_f0_bounds) or f0 > max(this_f0_bounds):
        print(f'f0 = {f0:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    t_envy      = np.zeros((nblocks, ntrials,))
    t_beta      = np.zeros((nblocks, ntrials,))
    t_norm_pe   = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        # RW norm update
        norms = RW(f0, lr, offers[b,:])
        all_norms[b,:] = norms.copy()

        for t in range(ntrials):

            # compute EV
            ev[b, t, 0] = offers[b,t] # accept (0)
            ev[b, t, 1] = envy * np.max(all_norms[b,t+1] - offers[b,t], 0) # reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A': #accept
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t])) #using value difference to calculate probability of accepting
            ch_prob[b, t, 0] = prob_accept.copy() #accept
            ch_prob[b, t, 1] = 1 - prob_accept.copy() #reject
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])

            # trial level saving
            t_envy[b, t] = envy
            t_beta[b, t] = beta
            t_norm_pe[b, t] = all_norms[b, t+1] - offers[b,t]
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000  
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     't_envy'    : t_envy,
                     't_beta'    : t_beta,
                     't_norm_pe' : t_norm_pe,
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'     : negll,
                     'BIC'       : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_RW_fixf0b(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the basic RW model with fixed initial norm at 10 (or specified) to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    f0 = 10
    envy = norm2envy(params[0])
    lr   = norm2alpha(params[1])
    beta = 1

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        print(f'lr = {lr:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 3]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        # RW norm update
        norms = RW(f0, lr, offers[b,:], norm=False)
        all_norms[b,:] = norms.copy()

        for t in range(ntrials):
            # compute EV
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_RW_fixf0(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the basic RW model with fixed initial norm at 10 (or specified) to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    f0 = 10
    envy = norm2envy(params[0])
    lr   = norm2alpha(params[1])
    beta = norm2beta(params[2])

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        print(f'lr = {lr:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 3]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks): #if nblocks==1, use reversals
        # RW norm update
        norms = RW(f0, lr, offers[b,:], norm=False)
        all_norms[b,:] = norms.copy()

        for t in range(ntrials):
            # compute EV
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict
    
def fit_FS_varf0(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the Fehr-Schmidtrn model with variable initial norm to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    envy = norm2envy(params[0])
    beta = norm2beta(params[1])
    f0   = norm2f0(params[2]) #double check

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 3]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        print(f'beta = {beta:.3f} not in range')
        return 10000000
    this_f0_bounds = [0.1, 20]
    if f0 < min(this_f0_bounds) or f0 > max(this_f0_bounds):
        print(f'f0 = {f0:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        all_norms[b,:] = f0 #double check here 

        for t in range(ntrials):
            # compute expected value (EV) and value difference (VD)
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_FS_varf0_fixb(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the Fehr-Schmidtrn model with variable initial norm to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    envy = norm2envy(params[0])
    f0   = norm2f0(params[1]) 
    beta = 1

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_f0_bounds = [0.1, 20]
    if f0 < min(this_f0_bounds) or f0 > max(this_f0_bounds):
        print(f'f0 = {f0:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        all_norms[b,:] = f0 #double check here 

        for t in range(ntrials):
            # compute expected value (EV) and value difference (VD)
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_FS_varf0_noBeta(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the Fehr-Schmidtrn model with variable initial norm to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    envy = norm2envy(params[0])
    f0   = norm2f0(params[1]) #double check here

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_f0_bounds = [0.1, 20]
    if f0 < min(this_f0_bounds) or f0 > max(this_f0_bounds):
        print(f'f0 = {f0:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        all_norms[b,:] = f0 #double check here 

        for t in range(ntrials):
            # compute expected value (EV) and value difference (VD)
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # make choice
            choices_idx = np.argmax(ev[b, t, :])
            choices[b, t] = ['A', 'R'][choices_idx]

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

def fit_FS_fixf0(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the Fehr-Schmidtrn model with fixed initial norm at 10 to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    f0 = 10
    envy = norm2envy(params[0])
    beta = norm2beta(params[1])

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 3]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        print(f'beta = {beta:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        all_norms[b,:] = f0 #double check here 

        for t in range(ntrials):
            # compute expected value (EV) and value difference (VD)
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

################################ Other older models ########################################
def fit_RW_f0normed(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the basic RW model with normed fix initial norm at 10 to a single subject's data. [exploration version, not really useful]
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial (what is this about?)
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)   
    envy = norm2envy(params[0])
    lr   = norm2alpha(params[1])
    beta = 1
    f0 = 10 # 10 even split percentage 

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_alpha_bounds = [0, 1]
    if lr < min(this_alpha_bounds) or lr > max(this_alpha_bounds):
        print(f'lr = {lr:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        norms = RW(f0, lr, offers[b,:], norm=True) #normed offers by percentage using total max offer 
        all_norms[b,:] = norms.copy()

        for t in range(ntrials):

            # compute EV
            ev[b, t, 0] = offers[b,t] # accept (0)
            ev[b, t, 1] = envy * np.max(all_norms[b,t+1] - offers[b,t], 0) # reject
            vd[b, t]  = offers[b, t] - envy * np.max((all_norms[b, t+1]*20) - offers[b,t], 0) # accept minus reject, convert norms back to offer space 

            # get choice index
            if choices[b, t] == 'A': #accept
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0
            
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t])) #using value difference to calculate probability of accepting
            ch_prob[b, t, 0] = prob_accept.copy() #accept
            ch_prob[b, t, 1] = 1 - prob_accept.copy() #reject
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict
def fit_B_varf0(params, choices, offers, prior=None, output='npl'):
    ''' 
    Fit the Bayesian norm adaption model with fixed initial norm at 10 (or specified) to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    envy = norm2envy(params[0])
    beta = norm2beta(params[1])
    f0   = norm2f0(params[2]) #double check
    k = 4 #copied from Matlab version
    # initial k and variance for prior 
    # need to initialize prior, uniform distribution, or use f0 to initiate that with variance (estimate initla variance around that)

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 3]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        print(f'beta = {beta:.3f} not in range')
        return 10000000
    this_f0_bounds = [0.1, 20]
    if f0 < min(this_f0_bounds) or f0 > max(this_f0_bounds):
        print(f'f0 = {f0:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        all_norms[b,:] = f0 #double check here 

        for t in range(ntrials):
            # bayesian norm update 
            k = k+1
            all_norms[b, t+1] = (k-1) / k * all_norms[b,t] + 1 / k * offers[b,t] #copied from Matlab version, don't understand here

            # no need to do the k = k+1 here 
            all_norms[b, t+1] = all_norms[b,t] + (1/(k+1)) * (offers[b,t] - all_norms[b,t]) # equation from the paper. 

            # bayesian update (other option but not the same as the one specified in paper)
            # post_t-1 = prior_t 
            # post_t = likeli_t * post_t-1

            # compute expected value (EV) and value difference (VD)
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict
def fit_B_fixf0(params, choices, offers, f0 = 10, prior=None, output='npl'):
    ''' 
    Fit the Bayesian norm adaption model with fixed initial norm at 10 (or specified) to a single subject's data.
        choices is a np.array with "A" or "R" for each trial
        rewards is a np.array with 1 (cue) or 0 (no cue) for each trial
        output is a string that specifies what to return (either 'nll' or 'all')
    '''
    nparams = len(params)
    envy = norm2envy(params[0])
    beta = norm2beta(params[1])
    k = 4 #copied from Matlab version

    # make sure params are in range
    this_envy_bounds = [0, 10]
    if envy < min(this_envy_bounds) or envy > max(this_envy_bounds):
        print(f'envy = {envy:.3f} not in range')
        return 10000000
    this_beta_bounds = [0.00001, 3]
    if beta < min(this_beta_bounds) or beta > max(this_beta_bounds):
        print(f'beta = {beta:.3f} not in range')
        return 10000000
    this_f0_bounds = [0.1, 20]
    if f0 < min(this_f0_bounds) or f0 > max(this_f0_bounds):
        print(f'f0 = {f0:.3f} not in range')
        return 10000000

    nblocks, ntrials = offers.shape

    ev          = np.zeros((nblocks, ntrials, 2))
    vd          = np.zeros((nblocks, ntrials,))
    ch_prob     = np.zeros((nblocks, ntrials,   2))
    choices_A   = np.zeros((nblocks, ntrials,))
    all_norms   = np.zeros((nblocks, ntrials+1,))
    rewards     = np.zeros((nblocks, ntrials,))
    choice_nll  = 0

    for b in range(nblocks):
        
        all_norms[b,:] = f0 #double check here 

        for t in range(ntrials):
            # bayesian norm update 
            k = k+1
            all_norms[b, t+1] = (k-1) / k * all_norms[b,t] + 1 / k * offers[b,t] #copied from Matlab version, don't understand here

            # compute expected value (EV) and value difference (VD)
            ev[b, t, :] = [offers[b,t], envy * np.max(all_norms[b,t+1] - offers[b,t], 0)] # accept, reject
            vd[b, t]  = offers[b, t] - envy * np.max(all_norms[b, t+1] - offers[b,t], 0) # accept minus reject

            # get choice index
            if choices[b, t] == 'A':
                c = 0
                choices_A[b, t] = 1
                rewards[b, t]   = offers[b,t].copy()
            else:
                c = 1
                choices_A[b, t] = 0
                rewards[b, t] = 0

            # calculate choice probability
            prob_accept = 1 / ( 1 + np.exp(-beta * vd[b, t]))
            ch_prob[b, t, 0] = prob_accept.copy()
            ch_prob[b, t, 1] = 1 - prob_accept.copy()
            
            # add to sum of choice nll for the block
            choice_nll += -np.log(ch_prob[b, t, c])
        
    # get the total negative log likelihood
    negll = choice_nll
    
    if output == 'npl':
        if prior is not None:  # EM-fit: P(Choices | h) * P(h | O) should be maximised, therefore same as minimizing it with negative sign
            fval = -(-negll + prior['logpdf'](params))            
            if np.isinf(fval):
                print(f'fval = {fval}')
                fval = 10000000
            return fval
        else: # NLL fit 
            return negll
        
    elif output == 'all':
        subj_dict = {'params'    : params,
                     'ev'        : ev, 
                     'vd'        : vd, 
                     'ch_prob'   : ch_prob, 
                     'choices'   : choices, 
                     'choices_A' : choices_A, 
                     'rewards'   : rewards, 
                     'offers'    : offers,
                     'norms'     : all_norms,
                     'negll'      : negll,
                     'BIC'        : nparams * np.log(ntrials*nblocks) + 2*negll}
        return subj_dict

