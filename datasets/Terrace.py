import os
import numpy as np
import re
from torchvision.datasets import VisionDataset
import math

class Terrace(VisionDataset):
    def __init__(self, root):
        super().__init__(root)
        # Terrace has xy-indexing: H*W=440*300, thus x is \in [0,300), y \in [0,440)
        # Terrace has consistent unit: milimeter (mm) for calibration & pos annotation
        self.__name__ = 'Terrace'
        self.img_shape, self.worldgrid_shape = [288, 360], [440, 300]
        # img_reduce 1 for too low resolution in Terrace
        self.img_reduce = 1

        self.num_cam, self.num_frame = 4, 4900
        self.indexing = 'xy'
        self.worldgrid2worldcoord_mat = np.array([[250, 0, 50], [250, 0, 150], [0, 0, 1]])
        self.intrinsic_matrices, self.extrinsic_matrices = zip(
            *[self.get_intrinsic_extrinsic_matrix(cam) for cam in range(self.num_cam)])
        # meter to millimeter
        self.unit = 1000

        worldgrid_translation = self.worldgrid2worldcoord_mat[:2, -1]
        self.worldgrid_translation = [ x1 - x2 * 0.025 * 100 // 2 for x1, x2 in zip(worldgrid_translation, self.worldgrid_shape[::-1]) ]

        # 0 for no expanded region with pedestrians in Terrace
        self.expand_width, self.expand_length = 0, 0
        # 0 for smaller region in Terrace, 4 for others
        self.bev_height, self.voxel_size, self.sigma = 30, 20, 0
        # if sigma <=0, default sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 = 0.8

    def get_image_fpaths(self, frame_range):
        img_fpaths = {cam: {} for cam in range(self.num_cam)}
        for camera_folder in sorted(os.listdir(os.path.join(self.root, 'Image_subsets'))):
            cam = int(camera_folder[-1]) - 1
            if cam >= self.num_cam:
                continue
            for fname in sorted(os.listdir(os.path.join(self.root, 'Image_subsets', camera_folder))):
                frame = int(fname.split('.')[0])
                if frame in frame_range:
                    img_fpaths[cam][frame] = os.path.join(self.root, 'Image_subsets', camera_folder, fname)
        return img_fpaths

    def get_worldgrid_from_pos(self, pos):
        grid_x = (pos % 30) * 10
        grid_y = (pos // 30) * 10
        return np.array([grid_x, grid_y], dtype=int)

    def get_pos_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        return (grid_x / 10) + (int(grid_y / 10 + 0.5) * 30)

    def get_worldgrid_from_worldcoord(self, world_coord):
        coord_x, coord_y = world_coord
        grid_x = ((coord_x + 500) / 250 - 0.5) * 10
        grid_y = ((coord_y + 1500) / 250 - 0.5) * 10
        return np.array([grid_x, grid_y], dtype=int)

    def get_worldcoord_from_worldgrid(self, worldgrid):
        grid_x, grid_y = worldgrid
        coord_x = (grid_x / 10 + 0.5) * 250 - 500
        coord_y = (grid_y / 10 + 0.5) * 250 - 1500
        return np.array([coord_x, coord_y])

    def get_worldcoord_from_pos(self, pos):
        grid = self.get_worldgrid_from_pos(pos)
        return self.get_worldcoord_from_worldgrid(grid)

    def get_pos_from_worldcoord(self, world_coord):
        grid = self.get_worldgrid_from_worldcoord(world_coord)
        return self.get_pos_from_worldgrid(grid)

    def get_intrinsic_extrinsic_matrix(self, camera_i):
        intrinsic_i, extrinsic_i = INTRINSIC[camera_i], EXTRINSIC[camera_i]
        intrinsic_matrix = np.array(
            [[intrinsic_i[0] / (DP * FRAME_SCALE), 0, intrinsic_i[2] / FRAME_SCALE],
             [0, intrinsic_i[0] / (DP * FRAME_SCALE), intrinsic_i[3] / FRAME_SCALE],
             [0, 0, 1]])
    
        rvec = np.array([extrinsic_i[3], extrinsic_i[4], extrinsic_i[5]])
        tvec = np.array([extrinsic_i[0], extrinsic_i[1], extrinsic_i[2]])
        
        rotation_matrix = np.array(internal_init(*rvec))
        translation_matrix = np.array(tvec, dtype=np.float32).reshape(3, 1) / 10
        extrinsic_matrix = np.hstack((rotation_matrix, translation_matrix))

        return intrinsic_matrix, extrinsic_matrix

    def read_pom(self):
        bbox_by_pos_cam = {}
        cam_pos_pattern = re.compile(r'(\d+) (\d+)')
        cam_pos_bbox_pattern = re.compile(r'(\d+) (\d+) ([-\d]+) ([-\d]+) (\d+) (\d+)')
        with open(os.path.join(self.root, 'rectangles.pom'), 'r') as fp:
            for line in fp:
                if 'RECTANGLE' in line:
                    cam, pos = map(int, cam_pos_pattern.search(line).groups())
                    if pos not in bbox_by_pos_cam:
                        bbox_by_pos_cam[pos] = {}
                    if 'notvisible' in line:
                        bbox_by_pos_cam[pos][cam] = None
                    else:
                        cam, pos, left, top, right, bottom = map(int, cam_pos_bbox_pattern.search(line).groups())
                        bbox_by_pos_cam[pos][cam] = [max(left, 0), max(top, 0),
                                                     min(right, 360 - 1), min(bottom, 288 - 1)]
        return bbox_by_pos_cam

def internal_init(mRx, mRy, mRz):
    sa = math.sin(mRx)
    ca = math.cos(mRx)
    sb = math.sin(mRy)
    cb = math.cos(mRy)
    sg = math.sin(mRz)
    cg = math.cos(mRz)

    mR11 = cb * cg
    mR12 = cg * sa * sb - ca * sg
    mR13 = sa * sg + ca * cg * sb
    mR21 = cb * sg
    mR22 = sa * sb * sg + ca * cg
    mR23 = ca * sb * sg - cg * sa
    mR31 = -sb
    mR32 = cb * sa
    mR33 = ca * cb
    
    return [[mR11, mR12, mR13], [mR21, mR22, mR23], [mR31, mR32, mR33]]

EXTRINSIC = [
    [-4.8441913843e+03, 5.5109448682e+02, 4.9667438357e+03, 1.9007833770e+00,
                                   4.9730769727e-01, 1.8415452559e-01],
    [-65.433635, 1594.811988, 2113.640844, 1.9347282363e+00, -7.0418616982e-01,
                                   -2.3783238362e-01],
    [1.9782813424e+03, -9.4027627332e+02, 1.2397750058e+04, -1.8289537286e+00,
                                   3.7748154985e-01, 3.0218614321e+00],
    [4.6737509054e+03, -2.5743341287e+01, 8.4155952460e+03, -1.8418460467e+00,
                                   -4.6728290805e-01, -3.0205552749e+00],
]

INTRINSIC = [
    [20.161920, 5.720865e-04, 366.514507, 305.832552, 1],
    [19.529144, 5.184242e-04, 360.228130, 255.166919, 1],
    [19.903218, 3.511557e-04, 355.506436, 241.205640, 1.0000000000e+00],
    [20.047015, 4.347668e-04, 349.154019, 245.786168, 1],
]

DP, FRAME_SCALE = 2.3e-02, 2