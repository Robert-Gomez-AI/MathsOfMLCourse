import numpy as np
import matplotlib.pyplot as plt

np.random.seed(72)

def flip(num_coins):
    coins = np.zeros(num_coins) #head: 1, tail: 0
    probs = np.random.uniform(size=num_coins)
    coins[probs > 0.5] = 1
    return coins

def flip_coins(num_coins,num_flips,show_freq=False):
    crand=np.random.choice(num_coins)
    total_hits= np.zeros(num_coins)

    for i in range(num_flips):
        total_hits+= flip(num_coins)
    
    frequency=total_hits/num_flips

    v1=frequency[0]
    vrand=frequency[crand]
    cmin = np.argmin(total_hits)
    vmin = frequency[cmin]
    
    if show_freq:
        print('Frequency of first coin: {}'.format(v1))
        print('Frequency and id of a random coin: id={},freq={}'.format(crand, vrand))
        print('Frequency and id of the coin with minimum frequency: id={},freq={}'.format(cmin, vmin))
    return v1,vrand,vmin

def run_experiment(num_coins,num_flips,iterations):
    
    v1s,vrands,vmins= np.zeros(iterations),np.zeros(iterations),np.zeros(iterations)
    for i in range(iterations):      
        v1s[i],vrands[i],vmins[i] = flip_coins(num_coins,num_flips)
    
    fig, axs = plt.subplots(1,3,sharey=True)
    fig.set_figheight(8)
    fig.set_figwidth(10)
    fig.suptitle('Distributions of $v$')
    n_bins = 10
    axs[0].hist(v1s,bins=n_bins)
    axs[0].set_title('$v_{1}$')
    axs[1].hist(vrands,bins=n_bins)
    axs[1].set_title('$v_{rands}$')
    axs[2].hist(vmins,bins=n_bins)
    axs[2].set_title('$v_{mins}$')
    return v1s,vrands,vmins 

def main():
    print('Utils.py is working')
if __name__=='__main__':
    main()