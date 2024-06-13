# Contains the functions used for create_nuscene_pkl_anno.ipynb file. It is the functions extracted from
# nuscene-devkit, but changed to return the variables related to 'camera_coordinates' and '3d_points'.
# You still need to 'pip install nuscene-devkit' as two classes here rely on other packages.
# Refer to original code https://github.com/nutonomy/nuscenes-devkit/tree/master/python-sdk
# Do not use invidually.

import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm


from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap


class NuScenes:
    """
    Database class for nuScenes to help query and retrieve information from the database.
    """

    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        """
        Loads database and creates reverse indexes and shortcuts.
        :param version: Version to load (e.g. "v1.0", ...).
        :param dataroot: Path to the tables and data.
        :param verbose: Whether to print status messages during load.
        :param map_resolution: Resolution of maps (meters).
        """
        self.version = version
        self.dataroot = dataroot
        self.verbose = verbose
        self.table_names = ['category', 'attribute', 'visibility', 'instance', 'sensor', 'calibrated_sensor',
                            'ego_pose', 'log', 'scene', 'sample', 'sample_data', 'sample_annotation', 'map']

        assert osp.exists(self.table_root), 'Database version not found: {}'.format(self.table_root)

        start_time = time.time()
        if verbose:
            print("======\nLoading NuScenes tables for version {}...".format(self.version))

        # Explicitly assign tables to help the IDE determine valid class members.
        self.category = self.__load_table__('category')
        self.attribute = self.__load_table__('attribute')
        self.visibility = self.__load_table__('visibility')
        self.instance = self.__load_table__('instance')
        self.sensor = self.__load_table__('sensor')
        self.calibrated_sensor = self.__load_table__('calibrated_sensor')
        self.ego_pose = self.__load_table__('ego_pose')
        self.log = self.__load_table__('log')
        self.scene = self.__load_table__('scene')
        self.sample = self.__load_table__('sample')
        self.sample_data = self.__load_table__('sample_data')
        self.sample_annotation = self.__load_table__('sample_annotation')
        self.map = self.__load_table__('map')

        # Initialize the colormap which maps from class names to RGB values.
        self.colormap = get_colormap()

        lidar_tasks = [t for t in ['lidarseg', 'panoptic'] if osp.exists(osp.join(self.table_root, t + '.json'))]
        if len(lidar_tasks) > 0:
            self.lidarseg_idx2name_mapping = dict()
            self.lidarseg_name2idx_mapping = dict()
            self.load_lidarseg_cat_name_mapping()
        for i, lidar_task in enumerate(lidar_tasks):
            if self.verbose:
                print(f'Loading nuScenes-{lidar_task}...')
            if lidar_task == 'lidarseg':
                self.lidarseg = self.__load_table__(lidar_task)
            else:
                self.panoptic = self.__load_table__(lidar_task)

            setattr(self, lidar_task, self.__load_table__(lidar_task))
            label_files = os.listdir(os.path.join(self.dataroot, lidar_task, self.version))
            num_label_files = len([name for name in label_files if (name.endswith('.bin') or name.endswith('.npz'))])
            num_lidarseg_recs = len(getattr(self, lidar_task))
            assert num_lidarseg_recs == num_label_files, \
                f'Error: there are {num_label_files} label files but {num_lidarseg_recs} {lidar_task} records.'
            self.table_names.append(lidar_task)
            # Sort the colormap to ensure that it is ordered according to the indices in self.category.
            self.colormap = dict({c['name']: self.colormap[c['name']]
                                  for c in sorted(self.category, key=lambda k: k['index'])})

        # If available, also load the image_annotations table created by export_2d_annotations_as_json().
        if osp.exists(osp.join(self.table_root, 'image_annotations.json')):
            self.image_annotations = self.__load_table__('image_annotations')

        # Initialize map mask for each map record.
        for map_record in self.map:
            map_record['mask'] = MapMask(osp.join(self.dataroot, map_record['filename']), resolution=map_resolution)

        if verbose:
            for table in self.table_names:
                print("{} {},".format(len(getattr(self, table)), table))
            print("Done loading in {:.3f} seconds.\n======".format(time.time() - start_time))

        # Make reverse indexes for common lookups.
        self.__make_reverse_index__(verbose)

        # Initialize NuScenesExplorer class.
        self.explorer = NuScenesExplorer(self)

    @property
    def table_root(self) -> str:
        """ Returns the folder where the tables are stored for the relevant version. """
        return osp.join(self.dataroot, self.version)

    def __load_table__(self, table_name) -> dict:
        """ Loads a table. """
        with open(osp.join(self.table_root, '{}.json'.format(table_name))) as f:
            table = json.load(f)
        return table

    def load_lidarseg_cat_name_mapping(self):
        """ Create mapping from class index to class name, and vice versa, for easy lookup later on """
        for lidarseg_category in self.category:
            # Check that the category records contain both the keys 'name' and 'index'.
            assert 'index' in lidarseg_category.keys(), \
                'Please use the category.json that comes with nuScenes-lidarseg, and not the old category.json.'

            self.lidarseg_idx2name_mapping[lidarseg_category['index']] = lidarseg_category['name']
            self.lidarseg_name2idx_mapping[lidarseg_category['name']] = lidarseg_category['index']

    def __make_reverse_index__(self, verbose: bool) -> None:
        """
        De-normalizes database to create reverse indices for common cases.
        :param verbose: Whether to print outputs.
        """

        start_time = time.time()
        if verbose:
            print("Reverse indexing ...")

        # Store the mapping from token to table index for each table.
        self._token2ind = dict()
        for table in self.table_names:
            self._token2ind[table] = dict()

            for ind, member in enumerate(getattr(self, table)):
                self._token2ind[table][member['token']] = ind

        # Decorate (adds short-cut) sample_annotation table with for category name.
        for record in self.sample_annotation:
            inst = self.get('instance', record['instance_token'])
            record['category_name'] = self.get('category', inst['category_token'])['name']

        # Decorate (adds short-cut) sample_data with sensor information.
        for record in self.sample_data:
            cs_record = self.get('calibrated_sensor', record['calibrated_sensor_token'])
            sensor_record = self.get('sensor', cs_record['sensor_token'])
            record['sensor_modality'] = sensor_record['modality']
            record['channel'] = sensor_record['channel']

        # Reverse-index samples with sample_data and annotations.
        for record in self.sample:
            record['data'] = {}
            record['anns'] = []

        for record in self.sample_data:
            if record['is_key_frame']:
                sample_record = self.get('sample', record['sample_token'])
                sample_record['data'][record['channel']] = record['token']

        for ann_record in self.sample_annotation:
            sample_record = self.get('sample', ann_record['sample_token'])
            sample_record['anns'].append(ann_record['token'])

        # Add reverse indices from log records to map records.
        if 'log_tokens' not in self.map[0].keys():
            raise Exception('Error: log_tokens not in map table. This code is not compatible with the teaser dataset.')
        log_to_map = dict()
        for map_record in self.map:
            for log_token in map_record['log_tokens']:
                log_to_map[log_token] = map_record['token']
        for log_record in self.log:
            log_record['map_token'] = log_to_map[log_record['token']]

        if verbose:
            print("Done reverse indexing in {:.1f} seconds.\n======".format(time.time() - start_time))

    def get(self, table_name: str, token: str) -> dict:
        """
        Returns a record from table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: Table record. See README.md for record details for each table.
        """
        assert table_name in self.table_names, "Table {} not found".format(table_name)

        return getattr(self, table_name)[self.getind(table_name, token)]

    def getind(self, table_name: str, token: str) -> int:
        """
        This returns the index of the record in a table in constant runtime.
        :param table_name: Table name.
        :param token: Token of the record.
        :return: The index of the record in table, table is an array.
        """
        return self._token2ind[table_name][token]

    def field2token(self, table_name: str, field: str, query) -> List[str]:
        """
        This function queries all records for a certain field value, and returns the tokens for the matching records.
        Warning: this runs in linear time.
        :param table_name: Table name.
        :param field: Field name. See README.md for details.
        :param query: Query to match against. Needs to type match the content of the query field.
        :return: List of tokens for the matching records.
        """
        matches = []
        for member in getattr(self, table_name):
            if member[field] == query:
                matches.append(member['token'])
        return matches

    def get_sample_data_path(self, sample_data_token: str) -> str:
        """ Returns the path to a sample_data. """

        sd_record = self.get('sample_data', sample_data_token)
        return osp.join(self.dataroot, sd_record['filename'])

    def get_sample_data(self, sample_data_token: str,
                        box_vis_level: BoxVisibility = BoxVisibility.ANY,
                        selected_anntokens: List[str] = None,
                        use_flat_vehicle_coordinates: bool = False) -> \
            Tuple[str, List[Box], np.array]:
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        :param sample_data_token: Sample_data token.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param selected_anntokens: If provided only return the selected annotation.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
                                             aligned to z-plane in the world.
        :return: (data_path, boxes, camera_intrinsic <np.array: 3, 3>)
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        cs_record = self.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = self.get('sensor', cs_record['sensor_token'])
        pose_record = self.get('ego_pose', sd_record['ego_pose_token'])

        data_path = self.get_sample_data_path(sample_data_token)

        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = None
            imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        if selected_anntokens is not None:
            boxes = list(map(self.get_box, selected_anntokens))
        else:
            boxes = self.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        box_list = []
        for box in boxes:
            if use_flat_vehicle_coordinates:
                # Move box to ego vehicle coord system parallel to world z plane.
                yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(scalar=np.cos(yaw / 2), vector=[0, 0, np.sin(yaw / 2)]).inverse)
            else:
                # Move box to ego vehicle coord system.
                box.translate(-np.array(pose_record['translation']))
                box.rotate(Quaternion(pose_record['rotation']).inverse)

                #  Move box to sensor coord system.
                box.translate(-np.array(cs_record['translation']))
                box.rotate(Quaternion(cs_record['rotation']).inverse)

            if sensor_record['modality'] == 'camera' and not \
                    box_in_image(box, cam_intrinsic, imsize, vis_level=box_vis_level):
                continue

            box_list.append(box)

        return data_path, box_list, cam_intrinsic

    def get_box(self, sample_annotation_token: str) -> Box:
        """
        Instantiates a Box class from a sample annotation record.
        :param sample_annotation_token: Unique sample_annotation identifier.
        """
        record = self.get('sample_annotation', sample_annotation_token)
        return Box(record['translation'], record['size'], Quaternion(record['rotation']),
                   name=record['category_name'], token=record['token'])

    def get_boxes(self, sample_data_token: str) -> List[Box]:
        """
        Instantiates Boxes for all annotation for a particular sample_data record. If the sample_data is a
        keyframe, this returns the annotations for that sample. But if the sample_data is an intermediate
        sample_data, a linear interpolation is applied to estimate the location of the boxes at the time the
        sample_data was captured.
        :param sample_data_token: Unique sample_data identifier.
        """

        # Retrieve sensor & pose records
        sd_record = self.get('sample_data', sample_data_token)
        curr_sample_record = self.get('sample', sd_record['sample_token'])

        if curr_sample_record['prev'] == "" or sd_record['is_key_frame']:
            # If no previous annotations available, or if sample_data is keyframe just return the current ones.
            boxes = list(map(self.get_box, curr_sample_record['anns']))

        else:
            prev_sample_record = self.get('sample', curr_sample_record['prev'])

            curr_ann_recs = [self.get('sample_annotation', token) for token in curr_sample_record['anns']]
            prev_ann_recs = [self.get('sample_annotation', token) for token in prev_sample_record['anns']]

            # Maps instance tokens to prev_ann records
            prev_inst_map = {entry['instance_token']: entry for entry in prev_ann_recs}

            t0 = prev_sample_record['timestamp']
            t1 = curr_sample_record['timestamp']
            t = sd_record['timestamp']

            # There are rare situations where the timestamps in the DB are off so ensure that t0 < t < t1.
            t = max(t0, min(t1, t))

            boxes = []
            for curr_ann_rec in curr_ann_recs:

                if curr_ann_rec['instance_token'] in prev_inst_map:
                    # If the annotated instance existed in the previous frame, interpolate center & orientation.
                    prev_ann_rec = prev_inst_map[curr_ann_rec['instance_token']]

                    # Interpolate center.
                    center = [np.interp(t, [t0, t1], [c0, c1]) for c0, c1 in zip(prev_ann_rec['translation'],
                                                                                 curr_ann_rec['translation'])]

                    # Interpolate orientation.
                    rotation = Quaternion.slerp(q0=Quaternion(prev_ann_rec['rotation']),
                                                q1=Quaternion(curr_ann_rec['rotation']),
                                                amount=(t - t0) / (t1 - t0))

                    box = Box(center, curr_ann_rec['size'], rotation, name=curr_ann_rec['category_name'],
                              token=curr_ann_rec['token'])
                else:
                    # If not, simply grab the current annotation.
                    box = self.get_box(curr_ann_rec['token'])

                boxes.append(box)
        return boxes

    def box_velocity(self, sample_annotation_token: str, max_time_diff: float = 1.5) -> np.ndarray:
        """
        Estimate the velocity for an annotation.
        If possible, we compute the centered difference between the previous and next frame.
        Otherwise we use the difference between the current and previous/next frame.
        If the velocity cannot be estimated, values are set to np.nan.
        :param sample_annotation_token: Unique sample_annotation identifier.
        :param max_time_diff: Max allowed time diff between consecutive samples that are used to estimate velocities.
        :return: <np.float: 3>. Velocity in x/y/z direction in m/s.
        """

        current = self.get('sample_annotation', sample_annotation_token)
        has_prev = current['prev'] != ''
        has_next = current['next'] != ''

        # Cannot estimate velocity for a single annotation.
        if not has_prev and not has_next:
            return np.array([np.nan, np.nan, np.nan])

        if has_prev:
            first = self.get('sample_annotation', current['prev'])
        else:
            first = current

        if has_next:
            last = self.get('sample_annotation', current['next'])
        else:
            last = current

        pos_last = np.array(last['translation'])
        pos_first = np.array(first['translation'])
        pos_diff = pos_last - pos_first

        time_last = 1e-6 * self.get('sample', last['sample_token'])['timestamp']
        time_first = 1e-6 * self.get('sample', first['sample_token'])['timestamp']
        time_diff = time_last - time_first

        if has_next and has_prev:
            # If doing centered difference, allow for up to double the max_time_diff.
            max_time_diff *= 2

        if time_diff > max_time_diff:
            # If time_diff is too big, don't return an estimate.
            return np.array([np.nan, np.nan, np.nan])
        else:
            return pos_diff / time_diff

    def get_sample_lidarseg_stats(self,
                                  sample_token: str,
                                  sort_by: str = 'count',
                                  lidarseg_preds_bin_path: str = None,
                                  gt_from: str = 'lidarseg') -> None:
        """
        Print the number of points for each class in the lidar pointcloud of a sample. Classes with have no
        points in the pointcloud will not be printed.
        :param sample_token: Sample token.
        :param sort_by: One of three options: count / name / index. If 'count`, the stats will be printed in
                        ascending order of frequency; if `name`, the stats will be printed alphabetically
                        according to class name; if `index`, the stats will be printed in ascending order of
                        class index.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param gt_from: 'lidarseg' or 'panoptic', ground truth source of point semantic labels.
        """
        assert gt_from in ['lidarseg', 'panoptic'], f'gt_from can only be lidarseg or panoptic, get {gt_from}'
        assert hasattr(self, gt_from), f'Error: You have no {gt_from} data; unable to get ' \
                                       'statistics for segmentation of the point cloud.'
        assert sort_by in ['count', 'name', 'index'], 'Error: sort_by can only be one of the following: ' \
                                                      'count / name / index.'
        semantic_table = getattr(self, gt_from)
        sample_rec = self.get('sample', sample_token)
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_record = self.get('sample_data', ref_sd_token)

        # Ensure that lidar pointcloud is from a keyframe.
        assert ref_sd_record['is_key_frame'], 'Error: Only pointclouds which are keyframes have ' \
                                              'lidar segmentation labels. Rendering aborted.'

        if lidarseg_preds_bin_path:
            lidarseg_labels_filename = lidarseg_preds_bin_path
            assert os.path.exists(lidarseg_labels_filename), \
                'Error: Unable to find {} to load the predictions for sample token {} ' \
                '(lidar sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, ref_sd_token)

            header = '===== Statistics for ' + sample_token + ' (predictions) ====='
        else:
            assert len(semantic_table) > 0, 'Error: There are no ground truth labels found for nuScenes-{} for {}.'\
                                            'Are you loading the test set? \nIf you want to see the sample statistics'\
                                            ' for your predictions, pass a path to the appropriate .bin/npz file using'\
                                            ' the lidarseg_preds_bin_path argument.'.format(gt_from, self.version)
            lidar_sd_token = self.get('sample', sample_token)['data']['LIDAR_TOP']
            lidarseg_labels_filename = os.path.join(self.dataroot,
                                                    self.get(gt_from, lidar_sd_token)['filename'])

            header = '===== Statistics for ' + sample_token + ' ====='
        print(header)

        points_label = load_bin_file(lidarseg_labels_filename, type=gt_from)
        if gt_from == 'panoptic':
            points_label = panoptic_to_lidarseg(points_label)
        lidarseg_counts = get_stats(points_label, len(self.lidarseg_idx2name_mapping))

        lidarseg_counts_dict = dict()
        for i in range(len(lidarseg_counts)):
            lidarseg_counts_dict[self.lidarseg_idx2name_mapping[i]] = lidarseg_counts[i]

        if sort_by == 'count':
            out = sorted(lidarseg_counts_dict.items(), key=lambda item: item[1])
        elif sort_by == 'name':
            out = sorted(lidarseg_counts_dict.items())
        else:
            out = lidarseg_counts_dict.items()

        for class_name, count in out:
            if count > 0:
                idx = self.lidarseg_name2idx_mapping[class_name]
                print('{:3}  {:40} n={:12,}'.format(idx, class_name, count))

        print('=' * len(header))

    def list_categories(self) -> None:
        self.explorer.list_categories()

    def list_lidarseg_categories(self, sort_by: str = 'count', gt_from: str = 'lidarseg') -> None:
        self.explorer.list_lidarseg_categories(sort_by=sort_by, gt_from=gt_from)

    def list_panoptic_instances(self, sort_by: str = 'count', get_hist: bool = False) -> None:
        self.explorer.list_panoptic_instances(sort_by=sort_by, get_hist=get_hist)

    def list_attributes(self) -> None:
        self.explorer.list_attributes()

    def list_scenes(self) -> None:
        self.explorer.list_scenes()

    def list_sample(self, sample_token: str) -> None:
        self.explorer.list_sample(sample_token)

    def render_pointcloud_in_image(self, sample_token: str, dot_size: int = 5, pointsensor_channel: str = 'LIDAR_TOP',
                                   camera_channel: str = 'CAM_FRONT', out_path: str = None,
                                   render_intensity: bool = False,
                                   show_lidarseg: bool = False,
                                   filter_lidarseg_labels: List = None,
                                   show_lidarseg_legend: bool = False,
                                   verbose: bool = True,
                                   lidarseg_preds_bin_path: str = None,
                                   show_panoptic: bool = False) -> None:
        pc_all, camera_coord, pc_cam_view, coloring, im = self.explorer.render_pointcloud_in_image(sample_token, dot_size, pointsensor_channel=pointsensor_channel,
                                                 camera_channel=camera_channel, out_path=out_path,
                                                 render_intensity=render_intensity,
                                                 show_lidarseg=show_lidarseg,
                                                 filter_lidarseg_labels=filter_lidarseg_labels,
                                                 show_lidarseg_legend=show_lidarseg_legend,
                                                 verbose=verbose,
                                                 lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                 show_panoptic=show_panoptic)

        return pc_all, camera_coord, pc_cam_view, coloring, im


class NuScenesExplorer:
    """ Helper class to list and visualize NuScenes data. These are meant to serve as tutorials and templates for
    working with the data. """

    def __init__(self, nusc: NuScenes):
        self.nusc = nusc

    def get_color(self, category_name: str) -> Tuple[int, int, int]:
        """
        Provides the default colors based on the category names.
        This method works for the general nuScenes categories, as well as the nuScenes detection categories.
        """

        return self.nusc.colormap[category_name]

    def list_categories(self) -> None:
        """ Print categories, counts and stats. These stats only cover the split specified in nusc.version. """
        print('Category stats for split %s:' % self.nusc.version)

        # Add all annotations.
        categories = dict()
        for record in self.nusc.sample_annotation:
            if record['category_name'] not in categories:
                categories[record['category_name']] = []
            categories[record['category_name']].append(record['size'] + [record['size'][1] / record['size'][0]])

        # Print stats.
        for name, stats in sorted(categories.items()):
            stats = np.array(stats)
            print('{:27} n={:5}, width={:5.2f}\u00B1{:.2f}, len={:5.2f}\u00B1{:.2f}, height={:5.2f}\u00B1{:.2f}, '
                  'lw_aspect={:5.2f}\u00B1{:.2f}'.format(name[:27], stats.shape[0],
                                                         np.mean(stats[:, 0]), np.std(stats[:, 0]),
                                                         np.mean(stats[:, 1]), np.std(stats[:, 1]),
                                                         np.mean(stats[:, 2]), np.std(stats[:, 2]),
                                                         np.mean(stats[:, 3]), np.std(stats[:, 3])))

    def list_lidarseg_categories(self, sort_by: str = 'count', gt_from: str = 'lidarseg') -> None:
        """
        Print categories and counts of the lidarseg data. These stats only cover
        the split specified in nusc.version.
        :param sort_by: One of three options: count / name / index. If 'count`, the stats will be printed in
                        ascending order of frequency; if `name`, the stats will be printed alphabetically
                        according to class name; if `index`, the stats will be printed in ascending order of
                        class index.
        :param gt_from: 'lidarseg' or 'panoptic', ground truth source of point semantic labels.
        """
        assert gt_from in ['lidarseg', 'panoptic'], f'gt_from can only be lidarseg or panoptic, get {gt_from}'
        assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'
        assert sort_by in ['count', 'name', 'index'], 'Error: sort_by can only be one of the following: ' \
                                                      'count / name / index.'

        print(f'Calculating semantic point stats for nuScenes-{gt_from}...')
        semantic_table = getattr(self.nusc, gt_from)
        start_time = time.time()

        # Initialize an array of zeroes, one for each class name.
        lidarseg_counts = [0] * len(self.nusc.lidarseg_idx2name_mapping)

        for record_lidarseg in semantic_table:
            lidarseg_labels_filename = osp.join(self.nusc.dataroot, record_lidarseg['filename'])
            points_label = load_bin_file(lidarseg_labels_filename, type=gt_from)
            if gt_from == 'panoptic':
                points_label = panoptic_to_lidarseg(points_label)

            indices = np.bincount(points_label)
            ii = np.nonzero(indices)[0]
            for class_idx, class_count in zip(ii, indices[ii]):
                lidarseg_counts[class_idx] += class_count

        lidarseg_counts_dict = dict()
        for i in range(len(lidarseg_counts)):
            lidarseg_counts_dict[self.nusc.lidarseg_idx2name_mapping[i]] = lidarseg_counts[i]

        if sort_by == 'count':
            out = sorted(lidarseg_counts_dict.items(), key=lambda item: item[1])
        elif sort_by == 'name':
            out = sorted(lidarseg_counts_dict.items())
        else:
            out = lidarseg_counts_dict.items()

        # Print frequency counts of each class in the lidarseg dataset.
        total_count = 0
        for class_name, count in out:
            idx = self.nusc.lidarseg_name2idx_mapping[class_name]
            print('{:3}  {:40} nbr_points={:12,}'.format(idx, class_name, count))
            total_count += count
        print('Calculated stats for {} point clouds in {:.1f} seconds, total {} points.\n====='.format(
            len(semantic_table), time.time() - start_time, total_count))

    def list_panoptic_instances(self, sort_by: str = 'count', get_hist: bool = False) -> None:
        """
        Print categories and counts of the lidarseg data. These stats only cover
        the split specified in nusc.version.
        :param sort_by: One of three options: count / name / index. If 'count`, the stats will be printed in
                        ascending order of frequency; if `name`, the stats will be printed alphabetically
                        according to class name; if `index`, the stats will be printed in ascending order of
                        class index.
        :param get_hist: True to return each frame' instance counts and per-category instance' number of frames, and
            number of points.
        """
        assert hasattr(self.nusc, 'panoptic'), f'Error: nuScenes-panoptic not installed!'
        assert sort_by in ['count', 'name', 'index'], 'Error: sort_by can only be one of the following: ' \
                                                      'count / name / index.'
        nusc_panoptic = getattr(self.nusc, 'panoptic')

        print(f'Calculating instance stats for nuScenes-panoptic ...')
        start_time = time.time()

        # {scene_token: np.ndarray((n, 5), np.int32)}, each row: (scene_id, frame_id, category_id, inst_id, num_points).
        scene_inst_stats = dict()
        for frame_id, record_panoptic in enumerate(nusc_panoptic):
            panoptic_label_filename = osp.join(self.nusc.dataroot, record_panoptic['filename'])
            panoptic_label = load_bin_file(panoptic_label_filename, type='panoptic')
            sample_token = self.nusc.get('sample_data', record_panoptic['sample_data_token'])['sample_token']
            scene_token = self.nusc.get('sample', sample_token)['scene_token']
            if scene_token not in scene_inst_stats:
                scene_inst_stats[scene_token] = np.empty((0, 4), dtype=np.int32)
            frame_cat_inst_count = get_frame_panoptic_instances(panoptic_label=panoptic_label, frame_id=frame_id)
            scene_inst_stats[scene_token] = np.append(scene_inst_stats[scene_token], frame_cat_inst_count, axis=0)

        panoptic_stats = get_panoptic_instances_stats(scene_inst_stats, self.nusc.lidarseg_idx2name_mapping, get_hist)
        pm = u"\u00B1"
        frame_num_insts = panoptic_stats['per_frame_panoptic_stats']['per_frame_num_instances']
        print('Per-frame number of instances: {:.0f}{}{:.0f}'.format(frame_num_insts[0], pm, frame_num_insts[1]))

        instance_counts = panoptic_stats['per_category_panoptic_stats'].copy()
        if sort_by == 'count':
            instance_counts = sorted(instance_counts.items(), key=lambda item: item[1]['num_instances'], reverse=True)
        elif sort_by == 'name':
            instance_counts = sorted(instance_counts.items())
        else:
            instance_counts = list(instance_counts.items())

        print('Per-category instance stats:')
        for cat_name, s in instance_counts:
            print('{}: {} instances, each instance spans to {:.0f}{}{:.0f} frames, with {:.0f}{}{:.0f} points'.format(
                cat_name, s['num_instances'], s['num_frames_per_instance'][0], pm, s['num_frames_per_instance'][1],
                s['num_points_per_instance'][0], pm, s['num_points_per_instance'][1]))

        num_instances, num_sample_annos = panoptic_stats['num_instances'], panoptic_stats['num_sample_annotations']
        print('\nCalculated stats for {} point clouds in {:.1f} seconds, total {} instances, {} sample annotations.'
              '\n====='.format(len(nusc_panoptic), time.time() - start_time, num_instances, num_sample_annos))

    def list_attributes(self) -> None:
        """ Prints attributes and counts. """
        attribute_counts = dict()
        for record in self.nusc.sample_annotation:
            for attribute_token in record['attribute_tokens']:
                att_name = self.nusc.get('attribute', attribute_token)['name']
                if att_name not in attribute_counts:
                    attribute_counts[att_name] = 0
                attribute_counts[att_name] += 1

        for name, count in sorted(attribute_counts.items()):
            print('{}: {}'.format(name, count))

    def list_scenes(self) -> None:
        """ Lists all scenes with some meta data. """

        def ann_count(record):
            count = 0
            sample = self.nusc.get('sample', record['first_sample_token'])
            while not sample['next'] == "":
                count += len(sample['anns'])
                sample = self.nusc.get('sample', sample['next'])
            return count

        recs = [(self.nusc.get('sample', record['first_sample_token'])['timestamp'], record) for record in
                self.nusc.scene]

        for start_time, record in sorted(recs):
            start_time = self.nusc.get('sample', record['first_sample_token'])['timestamp'] / 1000000
            length_time = self.nusc.get('sample', record['last_sample_token'])['timestamp'] / 1000000 - start_time
            location = self.nusc.get('log', record['log_token'])['location']
            desc = record['name'] + ', ' + record['description']
            if len(desc) > 55:
                desc = desc[:51] + "..."
            if len(location) > 18:
                location = location[:18]

            print('{:16} [{}] {:4.0f}s, {}, #anns:{}'.format(
                desc, datetime.utcfromtimestamp(start_time).strftime('%y-%m-%d %H:%M:%S'),
                length_time, location, ann_count(record)))

    def list_sample(self, sample_token: str) -> None:
        """ Prints sample_data tokens and sample_annotation tokens related to the sample_token. """

        sample_record = self.nusc.get('sample', sample_token)
        print('Sample: {}\n'.format(sample_record['token']))
        for sd_token in sample_record['data'].values():
            sd_record = self.nusc.get('sample_data', sd_token)
            print('sample_data_token: {}, mod: {}, channel: {}'.format(sd_token, sd_record['sensor_modality'],
                                                                       sd_record['channel']))
        print('')
        for ann_token in sample_record['anns']:
            ann_record = self.nusc.get('sample_annotation', ann_token)
            print('sample_annotation_token: {}, category: {}'.format(ann_record['token'], ann_record['category_name']))

    def map_pointcloud_to_image(self,
                                pointsensor_token: str,
                                camera_token: str,
                                min_dist: float = 1.0,
                                render_intensity: bool = False,
                                show_lidarseg: bool = False,
                                filter_lidarseg_labels: List = None,
                                lidarseg_preds_bin_path: str = None,
                                show_panoptic: bool = False) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :param min_dist: Distance from the camera below which points are discarded.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidar intensity instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """

        cam = self.nusc.get('sample_data', camera_token)
        pointsensor = self.nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            if show_lidarseg or show_panoptic:
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                # Ensure that lidar pointcloud is from a keyframe.
                assert pointsensor['is_key_frame'], \
                    'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                assert not render_intensity, 'Error: Invalid options selected. You can only select either ' \
                                             'render_intensity or show_lidarseg, not both.'

            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform from ego to the global frame.
        poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
        poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform from ego into the camera.
        cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        if render_intensity:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                              'not %s!' % pointsensor['sensor_modality']
            # Retrieve the color from the intensities.
            # Performs arbitary scaling to achieve more visually pleasing results.
            intensities = pc.points[3, :]
            intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
            intensities = intensities ** 0.1
            intensities = np.maximum(0, intensities - 0.5)
            coloring = intensities
        elif show_lidarseg or show_panoptic:
            assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render lidarseg labels for lidar, ' \
                                                              'not %s!' % pointsensor['sensor_modality']

            gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
            semantic_table = getattr(self.nusc, gt_from)

            if lidarseg_preds_bin_path:
                sample_token = self.nusc.get('sample_data', pointsensor_token)['sample_token']
                lidarseg_labels_filename = lidarseg_preds_bin_path
                assert os.path.exists(lidarseg_labels_filename), \
                    'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                    'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, pointsensor_token)
            else:
                if len(semantic_table) > 0:  # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                    lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                        self.nusc.get(gt_from, pointsensor_token)['filename'])
                else:
                    lidarseg_labels_filename = None

            if lidarseg_labels_filename:
                # Paint each label in the pointcloud with a RGBA value.
                if show_lidarseg:
                    coloring = paint_points_label(lidarseg_labels_filename,
                                                  filter_lidarseg_labels,
                                                  self.nusc.lidarseg_name2idx_mapping,
                                                  self.nusc.colormap)
                else:
                    coloring = paint_panop_points_label(lidarseg_labels_filename,
                                                        filter_lidarseg_labels,
                                                        self.nusc.lidarseg_name2idx_mapping,
                                                        self.nusc.colormap)

            else:
                coloring = depths
                print(f'Warning: There are no lidarseg labels in {self.nusc.version}. Points will be colored according '
                      f'to distance from the ego vehicle instead.')
        else:
            # Retrieve the color from the depth.
            coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        cam_cood = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
        # casing for non-keyframes which are slightly out of sync.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > min_dist)
        mask = np.logical_and(mask, cam_cood[0, :] > 1)
        mask = np.logical_and(mask, cam_cood[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, cam_cood[1, :] > 1)
        mask = np.logical_and(mask, cam_cood[1, :] < im.size[1] - 1)
        cam_cood = cam_cood[:, mask]
        coloring = coloring[mask]
        pc_cam_view = pc.points[:, mask]

        return pc, cam_cood, pc_cam_view, coloring, im

    def render_pointcloud_in_image(self,
                                   sample_token: str,
                                   dot_size: int = 5,
                                   pointsensor_channel: str = 'LIDAR_TOP',
                                   camera_channel: str = 'CAM_FRONT',
                                   out_path: str = None,
                                   render_intensity: bool = False,
                                   show_lidarseg: bool = False,
                                   filter_lidarseg_labels: List = None,
                                   ax: Axes = None,
                                   show_lidarseg_legend: bool = False,
                                   verbose: bool = True,
                                   lidarseg_preds_bin_path: str = None,
                                   show_panoptic: bool = False):
        """
        Scatter-plots a pointcloud on top of image.
        :param sample_token: Sample token.
        :param dot_size: Scatter plot dot size.
        :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param out_path: Optional path to save the rendered figure to disk.
        :param render_intensity: Whether to render lidar intensity instead of point depth.
        :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
        :param ax: Axes onto which to render.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param verbose: Whether to display the image in a window.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        if show_lidarseg:
            show_panoptic = False
        sample_record = self.nusc.get('sample', sample_token)

        # Here we just grab the front camera and the point sensor.
        pointsensor_token = sample_record['data'][pointsensor_channel]
        camera_token = sample_record['data'][camera_channel]

        pc, cam_cood, pc_cam_view, coloring, im = self.map_pointcloud_to_image(pointsensor_token, camera_token,
                                                            render_intensity=render_intensity,
                                                            show_lidarseg=show_lidarseg,
                                                            filter_lidarseg_labels=filter_lidarseg_labels,
                                                            lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                                            show_panoptic=show_panoptic)

        # Init axes.
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(9, 16))
            if lidarseg_preds_bin_path:
                fig.canvas.set_window_title(sample_token + '(predictions)')
            else:
                fig.canvas.set_window_title(sample_token)
        else:  # Set title on if rendering as part of render_sample.
            ax.set_title(camera_channel)
        ax.imshow(im)
        ax.scatter(cam_cood[0, :], cam_cood[1, :], c=coloring, s=1)
        ax.axis('off')

        if verbose:
            plt.show()
        return pc, cam_cood, pc_cam_view, coloring, im
