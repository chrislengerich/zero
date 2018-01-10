import numpy as np

def linear(p0, p1, timesteps):
    # Linear projection:

    # Assumptions:
    # Constant velocity, no acceleration.
    velocity = p1 - p0
    predictions = np.zeros((timesteps,3))
    for i in range(timesteps):
        predictions[i, :] = p0 + i * velocity
    return predictions

if __name__ == "__main__":
    p0 = np.array([1,2,3])
    p1 = np.array([1,3,9])
    print linear(p0, p1, 5)
