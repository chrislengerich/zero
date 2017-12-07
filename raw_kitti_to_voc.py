import numpy as np
import cv2
import xml.etree.ElementTree as et

def _load_image(path):
    return cv2.imread(path)

def _load_3d_coordinates(path):

    # Load 3d coordinates from a tracklet file.
    tree = et.parse(path)
    root = tree.getroot()

    centroids = {}
    for item in root.find('tracklets'):
        if item.tag == "item":
            if item.find("objectType").text == "Car":
                first_frame = int(item.find("first_frame").text)
                frame = first_frame

                poses = item.find("poses")
                for i in poses.iter("item"):
                    x = float(i.find("tx").text)
                    y = float(i.find("ty").text)
                    z = float(i.find("tz").text)
                    if frame not in centroids:
                        centroids[frame] = []
                    centroids[frame].append( {"x": x, "y": y, "z": z} )
                    frame += 1
    return centroids

def _tracks_to_image_centroids(tracks, camera_matrix, velodyne_to_camera):
    centroids = {}
    for frame, val in tracks.items():
        centroids[frame] = []
        for v in val:
            v = np.array([v["x"], v["y"], v["z"], 1])
            image_coords = camera_matrix.dot(velodyne_to_camera.dot(v))
            image_coords /= image_coords[2]
            centroids[frame].append(image_coords)
    return centroids

def _load_camera_matrix(path):
    with open(path) as f:

        K = None
        R = None
        T = None

        lines = f.readlines()
        for l in lines:
            l = l.split()
            if l[0] == "P_rect_02:":
                assert len(l) == 13, l
                P = np.array([float(li) for li in l[1:]])
                P = P.reshape((3,4))
            if l[0] == "R_rect_02:":
                assert len(l) == 10, l
                R = np.array([float(li) for li in l[1:]])
                R = R.reshape((3,3))
    R_rect = np.eye(4)
    R_rect[0:3, 0:3] = R
    return P.dot(R_rect)

def _load_velodyne_matrix(path):
    with open(path) as f:
        lines = f.readlines()
        for l in lines:
            l = l.split()
            if l[0] == "R:":
                assert len(l) == 10, l
                R = np.array([float(li) for li in l[1:]])
                R = R.reshape(3,3)
            if l[0] == "T:":
                assert len(l) == 4, l
                T = np.array([float(li) for li in l[1:]])
        R_transform = np.eye(4)
        R_transform[0:3,0:3] = R
        T_transform = np.eye(4)
        T_transform[0:3,3] = T
    return R_transform.dot(T_transform)

if __name__ == "__main__":
    camera_matrix = _load_camera_matrix("/home/ubuntu/kitti/testing/2011_09_26/calib_cam_to_cam.txt")
    print(camera_matrix)
    velodyne_to_camera = _load_velodyne_matrix("/home/ubuntu/kitti/testing/2011_09_26/calib_velo_to_cam.txt")
    print(velodyne_to_camera)
    image = _load_image("/home/ubuntu/kitti/testing/2011_09_26/2011_09_26_drive_0017_sync/image_02/data/0000000000.png")
    print image.shape
    tracks = _load_3d_coordinates("/home/ubuntu/kitti/testing/2011_09_26/2011_09_26_drive_0017_sync/tracklet_labels.xml")
    print(tracks[0])

    image_coords_tracks = _tracks_to_image_centroids(tracks, camera_matrix, velodyne_to_camera)
    for (i, val) in image_coords_tracks.items():
        print(i)
        print(val)

