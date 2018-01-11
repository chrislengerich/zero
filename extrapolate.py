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

def linear_image(p_image, timesteps, loader, idx):
    # Given two image points, return their linear extrapolation in image coordinates, passing through 3d.
    assert len(p_image) == 2
    p_world = [loader.image_to_world(idx, p_im) for p_im in p_image]
    ext = linear(p_world[0], p_world[1], timesteps)
    ext_image = [loader.world_to_image(idx, c) for c in ext]
    return ext_image

def loss(predictions, extrapolations):
    return np.linalg.norm(np.array(predictions) - np.array(extrapolations))

if __name__ == "__main__":
    p0 = np.array([1,2,3])
    p1 = np.array([1,3,9])
    print linear(p0, p1, 3)

    predictions = np.array([[1,2,3],[1,3,9], [1,5,17]])
    print loss(predictions, linear(p0, p1, 3))
