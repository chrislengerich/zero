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

def loss(predictions, extrapolations):
    return np.linalg.norm(np.array(predictions) - np.array(extrapolations))

if __name__ == "__main__":
    p0 = np.array([1,2,3])
    p1 = np.array([1,3,9])
    print linear(p0, p1, 3)

    predictions = np.array([[1,2,3],[1,3,9], [1,5,17]])
    print loss(predictions, linear(p0, p1, 3))
