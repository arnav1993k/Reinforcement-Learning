import numpy as np
from matplotlib import pyplot as plt
def get_policy(state,nA,estimator,epsilon):
    A = np.ones(nA, dtype=float) * epsilon / nA
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
    A[best_action] += (1.0 - epsilon)
    return A

def plot(X,y,title,xlabel,ylabel,name):
    fig = plt.figure(figsize=(10,10))
    plt.plot(X,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    fig.savefig("./plots/"+name+".png")
# def run_episode(env,state):
