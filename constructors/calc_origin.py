import os
import numpy
import logging
import copy

class BaseProcess2Origin:
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        self.task = task
        self.gt = gt
        self.ans = ans
        self.intr = intr
        self.extr = extr
        self.w_img = w_img
        self.h_img = h_img
        self.logger = logger
    
    # define check_format() and process_to_origin() for children class
    def check_format(self):
        raise NotImplementedError

    def process_to_origin(self, item):
        raise NotImplementedError

    def process(self):
        self.logger.info(f"Converting {self.task} to origin.")
        self.logger.info(f"gt: {self.gt}")
        self.logger.info(f"ans: {self.ans}")
        if not self.check_format():
            self.logger.warning("Invalid format.")
            return self.gt, "Invalid format."
        for itm in self.gt:
            itm = self.process_to_origin(itm)
        for itm in self.ans:
            itm = self.process_to_origin(itm)
        self.logger.info(f"gt_origin: {self.gt}")
        self.logger.info(f"ans_origin: {self.ans}")
        self.logger.info("Convert ok!")
        return self.gt, self.ans

class ProcessBbox2D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "oriented_bbox_2d" not in itm or len(itm["oriented_bbox_2d"]) != 5:
                return False
        return True

    def process_to_origin(self, item):
        u, v, w, h, angle_rad = item["oriented_bbox_2d"]
        u = u * self.w_img / 1000
        v = v * self.h_img / 1000
        w = w * self.w_img / 1000
        h = h * self.h_img / 1000
        item["oriented_bbox_2d"] = [u, v, w, h, angle_rad]
        return item

class ProcessBbox3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "bbox_uvd" not in itm or len(itm["bbox_uvd"]) != 9:
                return False
        return True
    
    def process_to_origin(self, item):
        u, v, depth, x, y, z, r, p, yaw = item["bbox_uvd"]
        u = u * self.w_img / 1000
        v = v * self.h_img / 1000
        cx = (u - self.intr[0][2]) * depth / self.intr[0][0]
        cy = (v - self.intr[1][2]) * depth / self.intr[1][1]
        cz = depth
        item["bbox_uvd"] = [cx, cy, cz, x, y, z, r, p, yaw]
        return item

class ProcessCorrespondence2D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "point_2d" not in itm or len(itm["point_2d"]) != 2:
                return False
        return True

    def process_to_origin(self, item):
        x, y = item["point_2d"]
        x = x * self.w_img / 1000
        y = y * self.h_img / 1000
        item["point_2d"] = [x, y]
        return item

class ProcessCorrespondence3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "point_uvd" not in itm or len(itm["point_uvd"]) != 3:
                return False
        return True

    def process_to_origin(self, item):
        u, v, d = item["point_uvd"]
        u = u * self.w_img / 1000
        v = v * self.h_img / 1000
        x = (u - self.intr[0][2]) * d / self.intr[0][0]
        y = (v - self.intr[1][2]) * d / self.intr[1][1]
        item["point_uvd"] = [x, y, d]
        return item

class ProcessGrasp2D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "grasp_point_2d" not in itm or len(itm["grasp_point_2d"]) != 2:
                return False
        return True

    def process_to_origin(self, item):
        x, y = item["grasp_point_2d"]
        x = x * self.w_img / 1000
        y = y * self.h_img / 1000
        item["grasp_point_2d"] = [x, y]
        return item

class ProcessGrasp3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "grasp_point_uvd" not in itm or len(itm["grasp_point_uvd"]) != 3:
                return False
        return True

    def process_to_origin(self, item):
        u, v, d = item["grasp_point_uvd"]
        u = u * self.w_img / 1000
        v = v * self.h_img / 1000
        x = (u - self.intr[0][2]) * d / self.intr[0][0]
        y = (v - self.intr[1][2]) * d / self.intr[1][1]
        item["grasp_point_uvd"] = [x, y, d]
        return item

class ProcessObjectPoint2D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "center_2d" not in itm or len(itm["center_2d"]) != 2:
                return False
        return True

    def process_to_origin(self, item):
        x, y = item["center_2d"]
        x = x * self.w_img / 1000
        y = y * self.h_img / 1000
        item["center_2d"] = [x, y]
        return item

class ProcessObjectPoint3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "center_uvd" not in itm or len(itm["center_uvd"]) != 3:
                return False
        return True

    def process_to_origin(self, item):
        u, v, d = item["center_uvd"]
        u = u * self.w_img / 1000
        v = v * self.h_img / 1000
        x = (u - self.intr[0][2]) * d / self.intr[0][0]
        y = (v - self.intr[1][2]) * d / self.intr[1][1]
        item["center_uvd"] = [x, y, d]
        return item

class ProcessSinglePoint2D(BaseProcess2Origin):
    def check_format(self):
        for itm in self.ans:
            if "functional_point_2d" not in itm or len(itm["functional_point_2d"]) != 2:
                return False
        return True

    def process_to_origin(self, item):
        x, y = item["functional_point_2d"]
        x = x * self.w_img / 1000
        y = y * self.h_img / 1000
        item["functional_point_2d"] = [x, y]
        return item

class ProcessSinglePoint3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "functional_point_uvd" not in itm or len(itm["functional_point_uvd"]) != 3:
                if "functional_point_uv_depth" not in itm or len(itm["functional_point_uv_depth"]) != 3:
                    return False
        return True

    def process_to_origin(self, item):
        try:
            u, v, d = item["functional_point_uvd"]
            key = "functional_point_uvd"
        except:
            u, v, d = item["functional_point_uv_depth"]
            key = "functional_point_uv_depth"
        u = u * self.w_img / 1000
        v = v * self.h_img / 1000
        x = (u - self.intr[0][2]) * d / self.intr[0][0]
        y = (v - self.intr[1][2]) * d / self.intr[1][1]
        item[key] = [x, y, d]
        return item

class ProcessSingleVector2D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "functional_vector" not in itm or len(itm["functional_vector"]) != 2:
                return False
        return True

    def process_to_origin(self, item):
        return item

class ProcessSingleVector3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "functional_vector" not in itm or len(itm["functional_vector"]) != 3:
                return False
        return True

    def process_to_origin(self, item):
        return item

class ProcessSpatialDistance(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "distance_meters" not in itm or not isinstance(itm["distance_meters"], (int, float)):
                return False
        return True

    def process_to_origin(self, item):
        return item

class ProcessSpatialChoice(BaseProcess2Origin):
    [
      {
        "spatial_relation_choice": "A", 
        "label": "choice_label"
      }
    ],
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "spatial_relation_choice" not in itm or not isinstance(itm["spatial_relation_choice"], str):
                return False
        return True

    def process_to_origin(self, item):
        return item

class ProcessTrajectory2D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)
    
    def check_format(self):
        for itm in self.ans:
            if "2d_trajectory" not in itm or not isinstance(itm["2d_trajectory"], list):
                return False
            for point in itm["2d_trajectory"]:
                if not isinstance(point, list) or len(point) != 2:
                    return False
        return True

    def process_to_origin(self, item):
        for point in item["2d_trajectory"]:
            x, y = point
            x = x * self.w_img / 1000
            y = y * self.h_img / 1000
            point[0] = x
            point[1] = y
        return item

class ProcessTrajectory3D(BaseProcess2Origin):
    def __init__(self, task, gt, ans, intr, extr, w_img, h_img, logger):
        super().__init__(task, gt, ans, intr, extr, w_img, h_img, logger)

    def check_format(self):
        for itm in self.ans:
            if "uvd_trajectory" not in itm or not isinstance(itm["uvd_trajectory"], list):
                return False
            for point in itm["uvd_trajectory"]:
                if not isinstance(point, list) or len(point) != 3:
                    return False
        return True

    def process_to_origin(self, item):
        for point in item["uvd_trajectory"]:
            u, v, d = point
            u = u * self.w_img / 1000
            v = v * self.h_img / 1000
            x = (u - self.intr[0][2]) * d / self.intr[0][0]
            y = (v - self.intr[1][2]) * d / self.intr[1][1]
            point[0] = x
            point[1] = y
        return item

def process_trajectory_3d(task, gt, ans, intr, extr, w_img, h_img, logger):
    pass

TASKS_CALC = {
    '2d_oriented_bbox_qa_detect_category': ProcessBbox2D,
    '2d_oriented_bbox_qa_single_part_with_desc': ProcessBbox2D,
    '2d_oriented_bbox_qa_single_with_desc': ProcessBbox2D,
    '3d_bbox_qa_single_part_with_desc': ProcessBbox3D,
    'obj_bbox_qa_detect_category_3d': ProcessBbox3D,
    'obj_bbox_qa_single_with_desc_3d': ProcessBbox3D,
    'correspondence_pairs_customized_2d': ProcessCorrespondence2D,
    'correspondence_pairs_customized_3d': ProcessCorrespondence3D,
    'correspondence_pairs_single_object_2d': ProcessCorrespondence2D,
    'correspondence_pairs_single_object_3d': ProcessCorrespondence3D,
    'grasp_point_2d': ProcessGrasp2D,
    'grasp_point_3d': ProcessGrasp3D,
    'object_point_qa_detect_category_2d': ProcessObjectPoint2D,
    'object_point_qa_detect_category_3d': ProcessObjectPoint3D,
    'object_point_qa_single_with_desc_2d': ProcessObjectPoint2D,
    'object_point_qa_single_with_desc_3d': ProcessObjectPoint3D,
    'single_func_point_2d': ProcessSinglePoint2D,
    'single_func_point_3d': ProcessSinglePoint3D,
    'single_func_vector_2d': ProcessSingleVector2D,
    'single_func_vector_3d': ProcessSingleVector3D,
    'single_part_point_2d': ProcessSinglePoint2D,
    'single_part_point_3d': ProcessSinglePoint3D,
    'spatial_rel_qa_obj2cam_distance': ProcessSpatialDistance,
    'spatial_rel_qa_obj2obj_choice': ProcessSpatialChoice,
    'spatial_rel_qa_obj2obj_distance': ProcessSpatialDistance,
    'traj_qa_prediction_2d': ProcessTrajectory2D,
    'traj_qa_prediction_3d': ProcessTrajectory3D,
}

def calc_origin_ans(task, gt_f, ans_f, intr, extr, w_img, h_img, logger):
    logger.info("Start calculating origin answers...")
    gt = copy.deepcopy(gt_f)
    ans = copy.deepcopy(ans_f)
    if not intr or not extr or intr == []:
        logger.warning("Not intr or extr")
        return gt, "Not intr or extr"
    
    if not isinstance(gt, list):
        gt = [gt]
    if not isinstance(ans, list):
        ans = [ans]
    
    if task in TASKS_CALC:
        process_class = TASKS_CALC[task]
        processor = process_class(task, gt, ans, intr, extr, w_img, h_img, logger)
        gt, ans = processor.process()
        return gt, ans
    return gt, "Task name invalid"

if __name__ == "__main__":
    test_gt = [
        {
            "oriented_bbox_2d": [
                285.6775,
                329.4948,
                101.3739,
                31.7539,
                1.5869
            ],
            "label": "boxed_beverage"
        }
    ]
    test_ans = [
        {
            "oriented_bbox_2d": [
                285.6775,
                329.4948,
                101.3739,
                31.7539,
                1.5869
            ],
            "label": "boxed_beverage"
        }
    ]
    intr = [
        [1280, 0, 640],
        [0, 720, 360],
        [0, 0, 1]
    ]
    extr = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    w_img = 1280
    h_img = 720

    # logger directly output to terminal
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    gt, ans = calc_origin_ans('2d_oriented_bbox_qa_detect_category', test_gt, test_ans, intr, extr, w_img, h_img, logger)
    print(test_gt)
    print(gt)
    print(test_ans)
    print(ans)