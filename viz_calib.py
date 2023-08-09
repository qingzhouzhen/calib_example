import numpy as np
import open3d as o3d
import pandas as pd
import collections

import math
import pickle
import os
import pdb

# check localization pose
class VisPCD(object):
    def __init__(self, window_name='pcd', bg_color=[255, 255, 255], point_size=2, color_map=None):
        self.window_name = window_name
        self.bg_color = bg_color
        self.point_size = point_size

        self.color_map = np.array([
            [50,0,255,255],
            [70,0,0,255],
            [90,0,0,255],
            [100,0,0,255],
            [120,0,0,255],
            [50,100,0,255],
            [60,110,0,255],
            [70,120,0,255],
            [80,130,0,255],
            [90,140,0,255],
            [100,150,0,255],
            [110,160,0,255],
            [120,170,0,255],
            [130,180,0,255],
            [140,190,0,255],
            [0,150,0,255],
            [0,160,0,255],
            [0,170,0,255],
            [0,190,0,255],
            [0,200,0,255],
            [0,0,255,255],])
        
        self.color_map = self.color_map[:, [2, 1, 0]]
        if color_map is not None:
            self.color_map = color_map
        
        self.lines = [[0,1],[1,5],[5,4],[4,0],
            [3,2],[2,6],[6,7],[7,3],
            [0,3],[4,7],[1,2],[5,6]]
        
        # set gt box color
        self.gt_colors = [[0, 255, 0] for i in range(len(self.lines))]
        for kn in [0, 1, 2, 3]:
            self.gt_colors[kn] = [255, 0, 0]
        
        # set pred box color
        self.pred_colors = [[0, 0, 255] for i in range(len(self.lines))]
        for kn in [0, 1, 2, 3]:
            self.pred_colors[kn] = [255, 0, 0]
        
    
    def _create_window(self):
        #axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame()

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=self.window_name)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray(self.bg_color)
        opt.point_size = self.point_size
        #opt.show_coordinate_frame = True
    
    def run(self, pcds, pcds_color=None, pcds_label=None, mask=None, gt_box3d_corners=None, is_clear=True, file_name=None, window_name = None):

        '''
        Input:
            pcds np.float64 array (n, c)
            pcds_color np.float64 array (n, 3)
            pcds_label np.long array (n,)
        '''
        if window_name:
            self.window_name = window_name
        self._create_window()
        if is_clear:
            self.vis.clear_geometries()
        
        # vis pcd
        pcd_l = o3d.geometry.PointCloud()
        pcd_l.points = o3d.utility.Vector3dVector(pcds[:, :3].astype(np.float64))
        if pcds_color is not None:
            pcd_l.colors = o3d.utility.Vector3dVector(pcds_color.astype(np.float64))
        elif pcds_label is not None:
            pcds_label_valid = pcds_label.copy()
            pcds_color_tmp = self.color_map[pcds_label_valid].astype(np.float64) / 255.0
            pcd_l.colors = o3d.utility.Vector3dVector(pcds_color_tmp)
        else:
            pcds_color_tmp = np.zeros((len(pcds), 3), dtype=np.float64)
            pcd_l.colors = o3d.utility.Vector3dVector(pcds_color_tmp)
        
        self.vis.add_geometry(pcd_l)
        # vis gt box
        if gt_box3d_corners is not None:
            for i in range(len(gt_box3d_corners)):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(gt_box3d_corners[i])
                
                line_set.lines = o3d.utility.Vector2iVector(self.lines)
                line_set.colors = o3d.utility.Vector3dVector(self.gt_colors)
                self.vis.add_geometry(line_set)
        
        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.run()
        #self.vis.destroy_window()
        if file_name is not None:
            self.vis.capture_screen_image(file_name)



def parse_pose_file(fname_pose):
    pose_dict = collections.OrderedDict()
    with open(fname_pose, 'r') as f:
        for line in f:
            s = line.strip().split(' ')
            pose_dict[s[0]] = np.array(s[1:], dtype=np.float32).reshape((4, 4))
    return pose_dict


def load_pcds(fname_pcd):
    pcds_xyzi = np.array(pd.read_csv(fname_pcd, sep=" ", skiprows=11, header=None)).astype(np.float32)
    valid_mask = ~np.isnan(pcds_xyzi.sum(axis=1))
    pcds_xyzi = pcds_xyzi[valid_mask]
    return pcds_xyzi


def Trans(pcds, mat):
    pcds_out = pcds.copy()
    
    pcds_tmp = pcds_out[:, :4].T
    pcds_tmp[-1] = 1
    pcds_tmp = mat.dot(pcds_tmp)
    pcds_tmp = pcds_tmp.T
    
    pcds_out[..., :3] = pcds_tmp[..., :3]
    pcds_out[..., 3:] = pcds[..., 3:]
    return pcds_out


def check_pose(base_dir,min_num = 90000):
    bag_list = os.listdir(base_dir)
    # bag_list = [_ for _ in bag_list if 'train' in _]
    bag_list.sort()
    vis = VisPCD()
    for bag_name in bag_list:
        print("bag name: ", bag_name)
        fname_pose = os.path.join(base_dir,  bag_name, "pose.txt")
        pose_dict = parse_pose_file(fname_pose)
        fpath_pcd = os.path.join(base_dir,  bag_name, '3d_url')
        pc_stack = []
        pc_stack_ori = []
        fn_list = [x for x in pose_dict.keys()]
        count = 0
        for t, fn in enumerate(fn_list):
            print("fn: ", fn)
            fname_pcd = os.path.join(fpath_pcd, '{}.pcd'.format(fn))
            pcds_xyzi = load_pcds(fname_pcd)
            if len(pcds_xyzi)< min_num:
                pcds_xyzi[:,-1]=20
            else:
                pcds_xyzi[:,-1]=t
            pcds_xyzi_trans = Trans(pcds_xyzi, pose_dict[fn])
            pcds_xyzi_trans_back = Trans(pcds_xyzi_trans, np.linalg.inv(pose_dict[fn_list[0]]))
            pc_stack_ori.append(pcds_xyzi_trans_back)
            pcds_xyzi_trans_back = pcds_xyzi_trans_back[np.where(pcds_xyzi_trans_back[:, 2] > 2)]
            pc_stack.append(pcds_xyzi_trans_back)
            count += 1
            if count == 2:
                break
        pc_stacked = np.concatenate(pc_stack_ori, axis=0)
        vis.run(pc_stacked ,pcds_label = pc_stacked[:,-1].astype(np.int), window_name = bag_name)
        # pose calib
        root = 'data'
        pcd_path1 = os.path.join(root, '1.pcd')
        pcd_path2 = os.path.join(root, '2.pcd')
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pc_stack[0][:,:3].astype(np.float64))
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pc_stack[1][:,:3].astype(np.float64))
        o3d.io.write_point_cloud(pcd_path1, pcd1)
        o3d.io.write_point_cloud(pcd_path2, pcd2)

        path = "calib/build/libpose_calib.so"
        from ctypes import cdll
        import ctypes
        mydll = cdll.LoadLibrary(path)
        file_name_in = pcd_path1.encode()
        file_name_trans = pcd_path2.encode()
        mydll.calib.restype = ctypes.POINTER(ctypes.c_double)
        c_array = mydll.calib(file_name_in, file_name_trans)
        return_array = np.array([c_array[i] for i in range(18)])
        print('return: ', return_array)
        trans_array = return_array[0:16].reshape((4,4))
        is_conv = return_array[-2]
        conf = return_array[-1]
        print('trans array: ', trans_array)
        print('is_conv: ', is_conv)
        print('score: ', conf)
        pcd0_trans = Trans(pc_stack_ori[0], trans_array)
        pc_stacked = np.concatenate((pcd0_trans, pc_stack_ori[1]), axis=0)
        vis.run(pc_stacked ,pcds_label = pc_stacked[:,-1].astype(np.int), window_name = bag_name)



if __name__ == '__main__':
    base_dir = 'data/bag'
    check_pose(base_dir)
