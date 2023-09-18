import cv2
import yaml
import numpy as np
import pickle
from dataclasses import dataclass
import open3d as o3d

from configs import *
class Camera:
    
    def __init__(self, yaml_file):
        with open(yaml_file, 'r') as file:
            loadeddict = yaml.load(file, Loader=yaml.FullLoader)
        def opencv_matrix(dict):
            mat = np.array(dict['data'])
            rows = dict['rows']
            cols = dict['cols']
            mat = mat.reshape(rows, cols)
            return mat
        self.cam_K = opencv_matrix(loadeddict.get('cam_K'))
        self.cam_kc = opencv_matrix(loadeddict.get('cam_kc'))
        self.proj_K = opencv_matrix(loadeddict.get('proj_K'))
        self.proj_kc = opencv_matrix(loadeddict.get('proj_kc'))
        self.R = opencv_matrix(loadeddict.get('R'))
        self.T = opencv_matrix(loadeddict.get('T'))
        self.cam_error = loadeddict.get('cam_error')
        self.proj_error = loadeddict.get('proj_error')
        self.stereo_error = loadeddict.get('stereo_error')
        self.cam_inv = np.linalg.inv(self.cam_K)
        self.proj_inv = np.linalg.inv(self.proj_K)

    # use the camera matrix and distortion coefficients to undistort the image
    def undistort(self, img):
        return cv2.undistort(img, self.cam_K, self.cam_kc)
    
    # Get the center of projector
    @property
    def projector_center(self):
        return -np.dot(self.R.T, self.T).flatten()
    
    # Get projector projection matrix
    @property
    def projector_projection_matrix(self):
        return np.dot(self.proj_K, np.hstack((self.R, self.T)))
    
    # Get camera projection matrix
    @property
    def camera_projection_matrix(self):
        return np.dot(self.cam_K, np.hstack((np.eye(3), np.zeros((3, 1)))))

    # Compute the camera ray for a given pixel
    def compute_camera_ray(self, p_cam):
        if len(p_cam) == 2:
            p_cam = np.hstack((p_cam, 1.0))
        return np.dot(self.cam_inv, p_cam)
    
    # Compute the projector plane normal for a given projector line
    def compute_plane_from_projector_line(self, l_proj_start, l_proj_end):
        # Convert line endpoints to rays in world coordinates
        ray_start = np.dot(self.proj_inv, np.array([l_proj_start[0], l_proj_start[1], 1.0]))
        ray_end = np.dot(self.proj_inv, np.array([l_proj_end[0], l_proj_end[1], 1.0]))

        # Compute a normal to the plane defined by the projector line
        n = np.cross(ray_start, ray_end)
        
        return n
    
    # Compute the intersection of a ray and a plane
    def intersect_ray_plane(self, O_cam, D_cam, n, P):
        denom = np.dot(n, D_cam)
        # Check if ray and plane are parallel
        if abs(denom) < 1e-6:
            return None
        t = - np.dot(n, (O_cam - P)) / denom    
        X = O_cam + t * D_cam
        return X

    

if __name__=="__main__":
    cam = Camera('configs/calibration.yml')

    with open('results/lines_decode_20230912-191834.pkl', 'rb') as f:
        lines = pickle.load(f)
        xx, yy, zz = [], [], []   
        for i, points in enumerate(lines):
            l_proj_start = [i * 10 - 540, -960]
            l_proj_end = [i * 10 -540, 960]
            n = cam.compute_plane_from_projector_line(l_proj_start, l_proj_end)
            
            for point in points:
                u, v = point
                
                # Compute camera ray for the pixel (u, v)
                p_cam = cam.compute_camera_ray([u - 1296 , v - 2304, 1])
                O_cam = np.array([0, 0, 0])  # Origin of the camera (since we're using normalized coordinates)
                D_cam = p_cam - O_cam  # Direction of the ray

                # Compute the intersection of the ray and the plane
                intersection_point = cam.intersect_ray_plane(O_cam, D_cam, n, cam.projector_center)
                if intersection_point is not None:
                    x, y, z = intersection_point
                    # Now (x, y, z) is the 3D position corresponding to the pixel (u, v).
                    print(f"Pixel ({u}, {v}) corresponds to 3D position ({x:.2f}, {y:.2f}, {z:.2f})")   
                    xx.append(x)
                    yy.append(y)
                    zz.append(z)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array([xx, yy, zz]).T)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud("results/3d_plot.ply", pcd)



    