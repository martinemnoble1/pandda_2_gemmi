import time
from pathlib import Path

import gemmi
import numpy as np

from ..interfaces import *
from .sparse_dmap import SparseDMap


class SparseDMapStream:
    def __init__(self,
                 datasets: Dict[str, DatasetInterface],
                 reference_frame: DFrameInterface,
                 alignments: Dict[str, AlignmentInterface],
                 # cache: Path,
                 transforms,
                 debug = False
                 ):
        self.datasets = datasets
        self.dframe = reference_frame
        self.alignments = alignments
        self.transforms = transforms
        self.debug = debug
        ...

    def load(self, dtag: str):
        dataset = self.datasets[dtag]
        alignment = self.alignments[dtag]

        begin_transform = time.time()
        for transform in self.transforms:
            dataset = transform(dataset)
        finish_transform = time.time()


        begin_fft = time.time()
        xmap = dataset.reflections.transform_f_phi_to_map()
        finish_fft = time.time()

        aligned_xmap = SparseDMapStream.align_xmap(xmap, self.dframe, alignment)

        return aligned_xmap

    @staticmethod
    def parallel_load(dataset, alignment, transforms, post_transforms, dframe, debug=False):

        begin = time.time()

        if debug:
            original_xmap = dataset.reflections.transform_f_phi_to_map(sample_rate=dataset.reflections.resolution()/0.4999)
            original_xmap_size = (original_xmap.nu, original_xmap.nv, original_xmap.nw)

        begin_transform = time.time()
        for transform in transforms:
            dataset = transform(dataset)
        finish_transform = time.time()
        # print(f"\tTransform: {finish_transform - begin_transform}")

        begin_fft = time.time()
        xmap = dataset.reflections.transform_f_phi_to_map(sample_rate=dataset.reflections.resolution()/0.4999)
        if debug:
            # print([x.label for x in dataset.reflections.reflections.columns])
            # print(np.array(dataset.reflections.reflections))
            arr = np.array(xmap)
            print(f'{dataset.name} raw xmap stats: min {np.min(arr)} max {np.max(arr)} mean {np.mean(arr)}')
            new_xmap_size = (xmap.nu, xmap.nv, xmap.nw)


        finish_fft = time.time()
        # print(f"\tFFT: {finish_fft - begin_fft}")

        aligned_xmap = SparseDMapStream.align_xmap(xmap, dframe, alignment)

        # Post transform
        for transform in post_transforms:
            transformed_xmap = transform(aligned_xmap, dframe, dataset.name)

        finish = time.time()

        if debug:
            transformed_xmap_size = (transformed_xmap.nu, transformed_xmap.nv, transformed_xmap.nw)
            print(f'Pre transform size: {original_xmap_size} vs transformed size: {new_xmap_size} vs pos-transformed: {transformed_xmap_size}')

        # print(f"Aligned xmap in: {round(finish-begin, 2)}")

        return SparseDMap.from_xmap(transformed_xmap, dframe, debug=debug)

    def array_load(self, ):
        # Get the shape to load datasets into
        shape = (len(self.datasets), self.dframe.mask.indicies[0].size)

        # Get the array (float32 -- the maps are f32; np.zeros defaults to f64
        # and would gratuitously double this datasets x voxels matrix).
        array = np.zeros(shape, dtype=np.float32)

        # Load each dataset in
        for j, dtag in enumerate(self.datasets):
            sparse_dmap = self.load(dtag)
            array[j, :] = sparse_dmap.data

        return array

    @staticmethod
    def align_xmap(xmap: CrystallographicGridInterface, dframe: DFrameInterface, alignment: AlignmentInterface):
        aligned_xmap = dframe.get_grid()

        # for residue_id in dframe.partitioning.partitions:
        #     point_position_array = dframe.partitioning.partitions[residue_id]
        #
        #     al = alignment.transforms[residue_id]
        #
        #     transform = al.get_transform()
        #     com_moving = al.com_moving
        #     com_reference = al.com_reference
        #
        #     points = point_position_array.points.astype(int).tolist()
        #     # print(f"Points length: {len(points)}")
        #     positions = point_position_array.positions.tolist()
        #
        #     gemmi.interpolate_points_single(
        #         xmap,
        #         aligned_xmap,
        #         points,
        #         positions,
        #         transform,
        #         com_moving,
        #         com_reference,
        #     )
        begin_listing = time.time()


        # transform_list = [alignment.transforms[residue_id].get_transform() for residue_id in
        #                   dframe.partitioning.partitions]
        # com_moving_list = [
        #     alignment.transforms[residue_id].com_moving.tolist()
        #     for residue_id
        #     in dframe.partitioning.partitions
        # ]
        # com_reference_list = [
        #     alignment.transforms[residue_id].com_reference.tolist()
        #     for residue_id in dframe.partitioning.partitions
        # ]
        transforms, com_ref, com_mov = alignment.get_transforms()
        # print(f"Partitions: {[key for key in dframe.partitioning.partitions]}")
        # print(f"Transforms: {[key for key in transforms]}")
        try:
            com_moving_list = [com_mov[residue_id].tolist() for residue_id in dframe.partitioning.partitions ]
            com_reference_list = [com_ref[residue_id].tolist() for residue_id in dframe.partitioning.partitions ]
            transform_list = [transforms[residue_id] for residue_id in dframe.partitioning.partitions ]
            for transform, com_m, com_r in zip(transform_list, com_moving_list, com_reference_list):
                transform.vec.fromlist(
                (gemmi.Vec3(*com_m) - transform.mat.multiply(gemmi.Vec3(*com_r))
                 ).tolist()
            )

            points_list = [dframe.partitioning.partitions[residue_id].points.astype(np.int32) for residue_id in
                           dframe.partitioning.partitions]
            positions_list = [dframe.partitioning.partitions[residue_id].positions.astype(np.float32) for residue_id in
                              dframe.partitioning.partitions]
        except Exception as e:
            print(e)
            transforms, com_ref, com_mov = alignment.get_transforms()
            print(f"Partitions: {[key for key in dframe.partitioning.partitions]}")
            print(f"Transforms: {[key for key in transforms]}")
            raise e

        # transforms, com_ref, com_mov = alignment.get_transforms()
        # transform_list = [transforms[residue_id] for residue_id in transforms]
        # com_moving_list = [com_mov[residue_id].tolist() for residue_id in transforms]
        # com_reference_list = [com_ref[residue_id].tolist() for residue_id in transforms]
        #
        # points_list = [np.copy(dframe.partitioning.partitions[residue_id].points) for residue_id in
        #                transforms]
        # positions_list = [np.copy(dframe.partitioning.partitions[residue_id].positions) for residue_id in
        #                   transforms]
        # print(f"\tListing: {finish_listing-begin_listing}")


        finish_listing = time.time()
        # print(f"\tListing: {finish_listing-begin_listing}")
        # print(com_reference_list)

        begin_interpolate = time.time()
        aligned_xmap.interpolate_grid_flexible(
            xmap,
            points_list,
            positions_list,
            transform_list,
        )
        finish_interpolate = time.time()
        # print(f"\tInterpolation: {finish_interpolate-begin_interpolate}")

        return aligned_xmap

    def __getitem__(self, dtag):
        ...
