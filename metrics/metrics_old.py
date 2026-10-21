"""Metric implementations and mapping for evaluation executor.

Provides:
- METRIC_BY_TASK: mapping from task name (or prefix) to metric key
- METRIC_IMPL: mapping metric key -> callable(gt, pred) -> float
- compute_metric(task_name, gt, pred) -> dict with metric_name and metric_value

The metric functions try to be tolerant to common ground-truth / prediction shapes: single point, list of points
or labeled points (dicts with 'label' and point fields). When possible we align by label, otherwise by index.
"""
from __future__ import annotations

import math
import json
import re
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import shapely for robust polygon intersection. If unavailable, we
# fall back to a conservative 0.0 IoU for oriented boxes.
try:
    from shapely.geometry import Polygon
    SHAPELY_AVAILABLE = True
except Exception:
    Polygon = None  # type: ignore
    SHAPELY_AVAILABLE = False

# ======================
# Helper Functions
# ======================
def _is_number_list(x) -> bool:
    if not isinstance(x, (list, tuple)):
        return False
    if len(x) == 0:
        return False
    return all(isinstance(v, (int, float)) for v in x)

def _euclidean(a: List[float], b: List[float]) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))))

def _cosine(a: List[float], b: List[float]) -> float:
    ax = [float(x) for x in a]
    bx = [float(x) for x in b]
    dot = sum(x * y for x, y in zip(ax, bx))
    na = math.sqrt(sum(x * x for x in ax))
    nb = math.sqrt(sum(x * x for x in bx))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))

def _normalize_label(lbl: Optional[str]) -> Optional[str]:
    if not lbl:
        return None
    s = str(lbl).lower()
    # strip trailing -active/_active/-passive/_passive
    import re
    s = re.sub(r"(?:[_-](?:active|passive))$", "", s)
    return s

def _extract_distance_from_qa(obj: Any) -> Optional[float]:
    """Try to find a numeric distance value (meters) inside QA-style objects.

    Handles these shapes (non-exhaustive):
    - numeric value (int/float)
    - dict with key 'distance_meters': 0.123
    - list of dicts like [ { 'distance_meters': 0.123, 'label': ... } ]
    - object with 'conversations' or 'value' fields that nest the above
    - string containing JSON with distance_meters
    - any nested combination of lists/dicts
    Returns first numeric distance found, or None.
    """
    if obj is None:
        return None

    # direct numeric
    if isinstance(obj, (int, float)):
        return float(obj)

    # string: try to parse JSON or extract numeric via regex
    if isinstance(obj, str):
        s = obj.strip()
        # try JSON parse
        try:
            parsed = json.loads(s)
        except Exception:
            # fallback: regex find first float-looking token
            m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", s)
            if m:
                try:
                    return float(m.group(1))
                except Exception:
                    return None
            return None
        return _extract_distance_from_qa(parsed)

    # dict: look for keys and recurse
    if isinstance(obj, dict):
        # common key
        if "distance_meters" in obj and isinstance(obj["distance_meters"], (int, float)):
            return float(obj["distance_meters"])
        # sometimes predictions are in 'value' or 'conversations'
        for k in ("value", "conversations", "answer", "answers"):
            if k in obj:
                return _extract_distance_from_qa(obj[k])
        # otherwise search all values
        for v in obj.values():
            res = _extract_distance_from_qa(v)
            if res is not None:
                return res
        return None

    # list/tuple: iterate and return first found
    if isinstance(obj, (list, tuple)):
        for item in obj:
            res = _extract_distance_from_qa(item)
            if res is not None:
                return res
        return None

    return None


# ======================
# BBOX Metrics
# ======================
def _extract_oriented_bboxes_from(obj: Any) -> List[Tuple[Optional[str], List[float]]]:
    """Extract oriented_bbox_2d entries as list of (label, [cx,cy,w,h,theta])"""
    out: List[Tuple[Optional[str], List[float]]] = []
    if obj is None:
        return out

    # If obj is a dict with key oriented_bbox_2d
    if isinstance(obj, dict):
        # direct key
        if "oriented_bbox_2d" in obj and _is_number_list(obj["oriented_bbox_2d"]):
            label = obj.get("label") or obj.get("name")
            return [(label, list(obj["oriented_bbox_2d"]))]
        # maybe a list under a value
        for v in obj.values():
            if _is_number_list(v) and len(v) == 5:
                return [(None, list(v))]

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if item is None:
                continue
            if isinstance(item, dict):
                if "oriented_bbox_2d" in item and _is_number_list(item["oriented_bbox_2d"]):
                    out.append((item.get("label") or item.get("name"), list(item["oriented_bbox_2d"])))
                    continue
                # fallback: any numeric list of length 5
                for vv in item.values():
                    if _is_number_list(vv) and len(vv) == 5:
                        out.append((item.get("label") or item.get("name"), list(vv)))
                        break
            elif _is_number_list(item) and len(item) == 5:
                out.append((None, list(item)))
        return out

    return out

def _extract_bbox3d_from(obj: Any) -> List[Tuple[Optional[str], List[float]]]:
    """Extract 3D axis-aligned bbox entries as list of (label, [cx,cy,cz,sx,sy,sz,...])

    Supports objects like:
    - {'bbox_3d': [cx,cy,cz,sx,sy,sz,...], 'label': 'name'}
    - [{'bbox_3d': [...], 'label': 'name'}, ...]
    - dicts where a value is a numeric list length >=6
    Returns list of (label, bbox_list)
    """
    out: List[Tuple[Optional[str], List[float]]] = []
    if obj is None:
        return out

    if _is_number_list(obj) and len(obj) >= 6:
        return [(None, list(obj))]

    if isinstance(obj, dict):
        # direct key
        if "bbox_3d" in obj and _is_number_list(obj["bbox_3d"]) and len(obj["bbox_3d"]) >= 6:
            label = obj.get("label") or obj.get("name")
            return [(label, list(obj["bbox_3d"]))]
        # fallback: scan values
        for vv in obj.values():
            if _is_number_list(vv) and len(vv) >= 6:
                return [(None, list(vv))]

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if item is None:
                continue
            if isinstance(item, dict):
                if "bbox_3d" in item and _is_number_list(item["bbox_3d"]) and len(item["bbox_3d"]) >= 6:
                    out.append((item.get("label") or item.get("name"), list(item["bbox_3d"])))
                    continue
                for vv in item.values():
                    if _is_number_list(vv) and len(vv) >= 6:
                        out.append((item.get("label") or item.get("name"), list(vv)))
                        break
            elif _is_number_list(item) and len(item) >= 6:
                out.append((None, list(item)))
        return out

    return out

def calculate_iou_2d_oriented(bbox1: List[float], bbox2: List[float]) -> float:
    """
    Compute IoU between two oriented 2D bboxes using Shapely polygons.

    bbox format: [cx, cy, w, h, theta]
    - theta is in radians (this matches the rest of the metrics code which
      represents rotation angles in radians).

    If Shapely is not available we return 0.0 as a conservative fallback.
    """
    if not SHAPELY_AVAILABLE:
        return 0.0

    def get_polygon(bbox):
        x, y, w, h, theta = bbox
        dx = w / 2.0
        dy = h / 2.0
        corners = np.array([
            [-dx, -dy],
            [dx, -dy],
            [dx, dy],
            [-dx, dy]
        ], dtype=float)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
        rotated = corners @ R.T
        rotated += np.array([x, y], dtype=float)
        return Polygon(rotated)

    try:
        
		
        poly1 = get_polygon(bbox1)
        poly2 = get_polygon(bbox2)
        print("poly1:", poly1)
        print("poly2:", poly2)
        if not poly1.is_valid or not poly2.is_valid:
            return 0.0
        inter = poly1.intersection(poly2).area
        union = poly1.area + poly2.area - inter
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0

# import math
# import numpy as np
# from shapely.geometry import Polygon
# from shapely.validation import explain_validity

# # 先确认SHAPELY_AVAILABLE的定义
# try:
#     from shapely.geometry import Polygon
#     from shapely.validation import explain_validity
#     SHAPELY_AVAILABLE = True
# except ImportError:
#     SHAPELY_AVAILABLE = False

# def calculate_iou_2d_oriented(bbox1: list[float], bbox2: list[float]) -> float:
#     """
#     Compute IoU between two oriented 2D bboxes using Shapely polygons.

#     bbox format: [cx, cy, w, h, theta]
#     - theta is in radians.
#     """
#     if not SHAPELY_AVAILABLE:
#         return 0.0

#     def get_polygon(bbox, name="bbox"):
#         x, y, w, h, theta = bbox
#         dx = w / 2.0
#         dy = h / 2.0
#         # 生成原始角点（以原点为中心）
#         corners = np.array([
#             [-dx, -dy],
#             [dx, -dy],
#             [dx, dy],
#             [-dx, dy]
#         ], dtype=float)
#         # 构建旋转矩阵
#         cos_t = math.cos(theta)
#         sin_t = math.sin(theta)
#         R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=float)
#         # 旋转并平移到中心
#         rotated = corners @ R.T
#         rotated += np.array([x, y], dtype=float)
#         # 生成多边形并打印调试信息
#         poly = Polygon(rotated)
#         print(f"=== {name} 多边形信息 ===")
#         print(f"原始角点: {corners}")
#         print(f"旋转矩阵: {R}")
#         print(f"旋转后角点: {rotated}")
#         print(f"多边形是否有效: {poly.is_valid}")
#         if not poly.is_valid:
#             print(f"无效原因: {explain_validity(poly)}")
#         print(f"多边形面积: {poly.area:.2f}")
#         print(f"多边形边界: {poly.bounds}\n")
#         return poly

#     try:
#         poly1 = get_polygon(bbox1, "GT")
#         poly2 = get_polygon(bbox2, "Pred")

#         # 检查多边形有效性
#         if not poly1.is_valid or not poly2.is_valid:
#             print("警告：多边形无效，返回IoU=0")
#             return 0.0

#         # 计算交集和并集
#         inter = poly1.intersection(poly2)
#         union = poly1.union(poly2)
#         inter_area = inter.area
#         union_area = union.area

#         # 打印交集/并集信息
#         print(f"交集面积: {inter_area:.6f}")
#         print(f"并集面积: {union_area:.6f}")
#         print(f"交集几何形状: {inter.geom_type}\n")

#         if union_area <= 1e-9:  # 避免除零
#             return 0.0
#         return inter_area / union_area

#     except Exception as e:
#         # 打印异常信息，不再吞掉错误
#         print(f"计算IoU时发生异常: {type(e).__name__}: {e}")
#         return 0.0

# def calculate_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
#     """Axis-aligned 3D IoU for bbox lists.

#     bbox format (accepted prefix): [cx, cy, cz, sx, sy, sz, ...]
#     Only the first 6 elements (center and size) are used.
#     """
#     try:
#         c1 = np.array(bbox1[:3], dtype=float)
#         s1 = np.array(bbox1[3:6], dtype=float)
#         c2 = np.array(bbox2[:3], dtype=float)
#         s2 = np.array(bbox2[3:6], dtype=float)
#     except Exception:
#         return 0.0

#     min1 = c1 - s1 / 2.0
#     max1 = c1 + s1 / 2.0
#     min2 = c2 - s2 / 2.0
#     max2 = c2 + s2 / 2.0

#     inter_min = np.maximum(min1, min2)
#     inter_max = np.minimum(max1, max2)
#     inter_size = np.maximum(inter_max - inter_min, 0.0)
#     inter_volume = float(inter_size[0] * inter_size[1] * inter_size[2])

#     vol1 = float(s1[0] * s1[1] * s1[2])
#     vol2 = float(s2[0] * s2[1] * s2[2])
#     union = vol1 + vol2 - inter_volume
#     if union <= 0:
#         return 0.0
#     return float(inter_volume / union)

# import numpy as np
# from scipy.spatial import ConvexHull

# ###############################################################
# # 工具函数：旋转矩阵
# ###############################################################
# def _rotation_matrix_3d(roll, pitch, yaw):
#     cr, sr = np.cos(roll), np.sin(roll)
#     cp, sp = np.cos(pitch), np.sin(pitch)
#     cy, sy = np.cos(yaw), np.sin(yaw)

#     Rz = np.array([[cy, -sy, 0],
#                    [sy,  cy, 0],
#                    [ 0,   0, 1]])

#     Ry = np.array([[ cp, 0, sp],
#                    [  0, 1,  0],
#                    [-sp, 0, cp]])

#     Rx = np.array([[1,  0,   0],
#                    [0, cr, -sr],
#                    [0, sr,  cr]])

#     return Rz @ Ry @ Rx


# ###############################################################
# # 工具函数：根据 box 参数生成 8 个顶点
# ###############################################################
# def _obb_vertices(box):
#     cx, cy, cz, sx, sy, sz, r, p, y = box
#     hx, hy, hz = sx / 2, sy / 2, sz / 2

#     # local corners
#     corners = np.array([
#         [ hx,  hy,  hz],
#         [ hx,  hy, -hz],
#         [ hx, -hy,  hz],
#         [ hx, -hy, -hz],
#         [-hx,  hy,  hz],
#         [-hx,  hy, -hz],
#         [-hx, -hy,  hz],
#         [-hx, -hy, -hz],
#     ])

#     R = _rotation_matrix_3d(r, p, y)
#     rotated = corners @ R.T
#     translated = rotated + np.array([cx, cy, cz])

#     return translated


# ###############################################################
# # 工具函数：凸包体积
# ###############################################################
# def _volume(points):
#     if points.shape[0] < 4:
#         return 0.0
#     try:
#         return ConvexHull(points).volume
#     except:
#         return 0.0


# ###############################################################
# # 工具：多面体求交
# ###############################################################
# def _intersect_polyhedron(box1, box2):
#     """
#     使用 pyhull 进行 halfspace intersection（如无 pyhull 会报错）
#     用户可自行安装: pip install pyhull
#     """
#     try:
#         from pyhull.halfspace import Halfspace, HalfspaceIntersection
#     except:
#         # 若 pyhull 不可用 → 返回无交点
#         return np.zeros((0, 3))

#     # 将一个凸多面体（由 8 个点构成）转换为半空间约束
#     def box_to_halfspaces(vertices):
#         hull = ConvexHull(vertices)
#         halfspaces = []
#         for eq in hull.equations:
#             normal = eq[:3]
#             offset = eq[3]  # normal·x + offset = 0 → <=0 是内部
#             halfspaces.append(Halfspace(np.hstack([normal, offset])))
#         interior = vertices.mean(axis=0)
#         return halfspaces, interior

#     v1 = _obb_vertices(box1)
#     v2 = _obb_vertices(box2)

#     h1, c1 = box_to_halfspaces(v1)
#     h2, c2 = box_to_halfspaces(v2)

#     all_hs = h1 + h2
#     interior_pt = (c1 + c2) / 2

#     try:
#         hs_inter = HalfspaceIntersection(all_hs, interior_pt)
#         return np.array(hs_inter.intersections)
#     except:
#         return np.zeros((0, 3))


# ###############################################################
# # 最终接口：与原来函数完全相同
# ###############################################################
# def calculate_iou_3d(bbox1, bbox2) -> float:
#     """
#     输入：bbox = [cx, cy, cz, sx, sy, sz, r, p, y]
#     输出：float（0~1）
#     """
#     # 保留原错误处理行为
#     try:
#         b1 = np.array(bbox1, dtype=float)
#         b2 = np.array(bbox2, dtype=float)
#     except:
#         return 0.0

#     # 体积
#     V1 = _volume(_obb_vertices(b1))
#     V2 = _volume(_obb_vertices(b2))

#     # 相交体积
#     inter_pts = _intersect_polyhedron(b1, b2)
#     VI = _volume(inter_pts)
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("VI:",VI)
#     if V1 + V2 - VI <= 0:
#         return 0.0
#     return float(VI / (V1 + V2 - VI))
# import numpy as np
# from scipy.spatial.transform import Rotation as R
# from typing import List

# def create_obb(center: np.ndarray, size: np.ndarray, rot_matrix: np.ndarray) -> np.ndarray:
#     """
#     创建有向包围盒（OBB）的8个顶点
#     Args:
#         center: 中心点 (cx, cy, cz)
#         size: 尺寸 (sx, sy, sz)
#         rot_matrix: 3x3 旋转矩阵
#     Returns:
#         8x3 的顶点坐标数组
#     """
#     # 局部坐标系下的轴对齐顶点（8个）
#     half_size = size / 2.0
#     local_vertices = np.array([
#         [-half_size[0], -half_size[1], -half_size[2]],
#         [-half_size[0], -half_size[1],  half_size[2]],
#         [-half_size[0],  half_size[1], -half_size[2]],
#         [-half_size[0],  half_size[1],  half_size[2]],
#         [ half_size[0], -half_size[1], -half_size[2]],
#         [ half_size[0], -half_size[1],  half_size[2]],
#         [ half_size[0],  half_size[1], -half_size[2]],
#         [ half_size[0],  half_size[1],  half_size[2]],
#     ])
    
#     # 旋转并平移到世界坐标系
#     world_vertices = (rot_matrix @ local_vertices.T).T + center
#     return world_vertices

# def compute_aabb(vertices: np.ndarray) -> tuple:
#     """计算顶点集的轴对齐包围盒（AABB）"""
#     min_coords = np.min(vertices, axis=0)
#     max_coords = np.max(vertices, axis=0)
#     return min_coords, max_coords

# def aabb_intersection_volume(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
#     """计算两个AABB的交集体积"""
#     inter_min = np.maximum(min1, min2)
#     inter_max = np.minimum(max1, max2)
#     inter_size = np.maximum(inter_max - inter_min, 0.0)
#     return inter_size[0] * inter_size[1] * inter_size[2]

# def calculate_iou_3d(bbox1: List[float], bbox2: List[float], degrees: bool = True) -> float:
#     """
#     计算3D有向包围盒（OBB）的IoU
#     输入格式：CX,CY,CZ,SX,SY,SZ,R,P,Y（r=roll, p=pitch, y=yaw）
    
#     Args:
#         bbox1: 第一个3D框参数列表
#         bbox2: 第二个3D框参数列表
#         degrees: 旋转角是否为角度制（默认True）
#     Returns:
#         3D IoU值（0~1）
#     """
#     try:
#         # 解析参数
#         c1 = np.array(bbox1[:3], dtype=float)  # 中心点
#         s1 = np.array(bbox1[3:6], dtype=float)  # 尺寸（边长）
#         r1, p1, y1 = bbox1[6:9]                # 旋转角（roll,pitch,yaw）
        
#         c2 = np.array(bbox2[:3], dtype=float)
#         s2 = np.array(bbox2[3:6], dtype=float)
#         r2, p2, y2 = bbox2[6:9]
        
#         # 验证尺寸有效性
#         if np.any(s1 <= 0) or np.any(s2 <= 0):
#             return 0.0
        
#     except (IndexError, ValueError):
#         return 0.0
    
#     # 1. 生成旋转矩阵（注意欧拉角顺序是xyz，对应roll,pitch,yaw）
#     rot1 = R.from_euler('xyz', [r1, p1, y1], degrees=degrees)
#     rot_matrix1 = rot1.as_matrix()
#     rot2 = R.from_euler('xyz', [r2, p2, y2], degrees=degrees)
#     rot_matrix2 = rot2.as_matrix()
    
#     # 2. 创建两个OBB的顶点
#     vertices1 = create_obb(c1, s1, rot_matrix1)
#     vertices2 = create_obb(c2, s2, rot_matrix2)
    
#     # 3. 计算AABB（快速预判断：如果AABB无交集，直接返回0）
#     min1, max1 = compute_aabb(vertices1)
#     min2, max2 = compute_aabb(vertices2)
#     aabb_inter_vol = aabb_intersection_volume(min1, max1, min2, max2)
#     if aabb_inter_vol <= 1e-8:
#         return 0.0
    
#     # 4. 计算两个OBB的体积（旋转不改变体积）
#     vol1 = s1[0] * s1[1] * s1[2]
#     vol2 = s2[0] * s2[1] * s2[2]
    
#     # 5. 计算OBB交集体积
#     inter_vol = aabb_inter_vol  # 若需高精度，替换为SAT计算的交集体积
    
#     # 6. 计算IoU
#     union_vol = vol1 + vol2 - inter_vol
#     if union_vol <= 1e-8:
#         return 0.0
    
#     return float(inter_vol / union_vol)

# import numpy as np

# def calculate_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
#     """3D IoU for oriented bbox lists with rotation angles (r,p,y).
    
#     bbox format: [cx, cy, cz, sx, sy, sz, r, p, y]
#     """
#     try:
#         c1 = np.array(bbox1[:3], dtype=float)  # Center of box1
#         s1 = np.array(bbox1[3:6], dtype=float)  # Size of box1 (sx, sy, sz)
#         rpy1 = np.radians(bbox1[6:])  # Rotation angles (r, p, y) in radians

#         c2 = np.array(bbox2[:3], dtype=float)  # Center of box2
#         s2 = np.array(bbox2[3:6], dtype=float)  # Size of box2 (sx, sy, sz)
#         rpy2 = np.radians(bbox2[6:])  # Rotation angles (r, p, y) in radians
#     except Exception:
#         return 0.0

#     def rotation_matrix(rpy):
#         """Creates a rotation matrix from the roll, pitch, yaw angles."""
#         r, p, y = rpy
#         R_x = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
#         R_y = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
#         R_z = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
#         return R_z @ R_y @ R_x  # Combined rotation matrix

#     # Compute the rotation matrix for each bbox
#     R1 = rotation_matrix(rpy1)
#     R2 = rotation_matrix(rpy2)

#     def get_rotated_corners(center, size, rotation_matrix):
#         """Get the 8 corners of the rotated bounding box."""
#         half_size = size / 2.0
#         # Define the 8 corner points of the axis-aligned bounding box
#         corners = np.array([
#             [half_size[0], half_size[1], half_size[2]], [-half_size[0], half_size[1], half_size[2]],
#             [-half_size[0], -half_size[1], half_size[2]], [half_size[0], -half_size[1], half_size[2]],
#             [half_size[0], half_size[1], -half_size[2]], [-half_size[0], half_size[1], -half_size[2]],
#             [-half_size[0], -half_size[1], -half_size[2]], [half_size[0], -half_size[1], -half_size[2]]
#         ])
#         # Apply rotation
#         rotated_corners = np.dot(corners, rotation_matrix.T)
#         # Translate to the center of the box
#         rotated_corners += center
#         return rotated_corners

#     # Get the rotated corners of both boxes
#     corners1 = get_rotated_corners(c1, s1, R1)
#     corners2 = get_rotated_corners(c2, s2, R2)

#     # Compute the AABB (Axis-Aligned Bounding Box) of the intersection of the two rotated boxes
#     min1 = np.min(corners1, axis=0)
#     max1 = np.max(corners1, axis=0)
#     min2 = np.min(corners2, axis=0)
#     max2 = np.max(corners2, axis=0)

#     inter_min = np.maximum(min1, min2)
#     inter_max = np.minimum(max1, max2)
#     inter_size = np.maximum(inter_max - inter_min, 0.0)
#     inter_volume = float(inter_size[0] * inter_size[1] * inter_size[2])

#     vol1 = float(s1[0] * s1[1] * s1[2])
#     vol2 = float(s2[0] * s2[1] * s2[2])
#     union_volume = vol1 + vol2 - inter_volume

#     if union_volume <= 0:
#         return 0.0
#     return float(inter_volume / union_volume)
# import numpy as np
# from typing import List, Tuple, Optional


# def euler_to_rotation_matrix(euler_angles: np.ndarray) -> np.ndarray:
#     """
#     将欧拉角（rx, ry, rz）转换为旋转矩阵（ZYX顺序，绕x→y→z轴旋转）
#     :param euler_angles: 欧拉角数组 [rx, ry, rz]，单位：弧度
#     :return: 3x3 旋转矩阵
#     """
#     rx, ry, rz = euler_angles
#     # 绕x轴旋转
#     R_x = np.array([
#         [1, 0, 0],
#         [0, np.cos(rx), -np.sin(rx)],
#         [0, np.sin(rx), np.cos(rx)]
#     ])
#     # 绕y轴旋转
#     R_y = np.array([
#         [np.cos(ry), 0, np.sin(ry)],
#         [0, 1, 0],
#         [-np.sin(ry), 0, np.cos(ry)]
#     ])
#     # 绕z轴旋转
#     R_z = np.array([
#         [np.cos(rz), -np.sin(rz), 0],
#         [np.sin(rz), np.cos(rz), 0],
#         [0, 0, 1]
#     ])
#     # 组合旋转矩阵（ZYX顺序）
#     R = np.dot(R_z, np.dot(R_y, R_x))
#     return R


# def get_obb_vertices(center: np.ndarray, size: np.ndarray, rot_mat: np.ndarray) -> np.ndarray:
#     """
#     生成3D OBB的8个顶点
#     :param center: 中心点 [cx, cy, cz]
#     :param size: 尺寸 [sx, sy, sz]（沿主轴的长度）
#     :param rot_mat: 3x3 旋转矩阵
#     :return: 8x3 的顶点数组
#     """
#     # OBB局部坐标系的8个顶点（中心在原点）
#     half_size = size / 2.0
#     local_vertices = np.array([
#         [half_size[0], half_size[1], half_size[2]],
#         [half_size[0], half_size[1], -half_size[2]],
#         [half_size[0], -half_size[1], half_size[2]],
#         [half_size[0], -half_size[1], -half_size[2]],
#         [-half_size[0], half_size[1], half_size[2]],
#         [-half_size[0], half_size[1], -half_size[2]],
#         [-half_size[0], -half_size[1], half_size[2]],
#         [-half_size[0], -half_size[1], -half_size[2]],
#     ])
#     # 旋转+平移到世界坐标系
#     world_vertices = np.dot(local_vertices, rot_mat.T) + center
#     return world_vertices


# def project_vertices_on_axis(vertices: np.ndarray, axis: np.ndarray) -> Tuple[float, float]:
#     """
#     将顶点投影到指定轴上，返回投影的最小/最大值
#     :param vertices: Nx3 顶点数组
#     :param axis: 3D 投影轴（单位向量）
#     :return: (投影最小值, 投影最大值)
#     """
#     axis = axis / np.linalg.norm(axis)  # 归一化轴
#     projections = np.dot(vertices, axis)
#     return float(np.min(projections)), float(np.max(projections))


# def sat_3d_obb_intersection(
#     vertices1: np.ndarray, 
#     rot_mat1: np.ndarray,
#     vertices2: np.ndarray, 
#     rot_mat2: np.ndarray
# ) -> bool:
#     """
#     3D分离轴定理（SAT）判断两个OBB是否相交
#     分离轴包括：
#     1. OBB1的3个主轴
#     2. OBB2的3个主轴
#     3. OBB1主轴 × OBB2主轴（共9个叉乘）
#     总计15个分离轴，只要有一个轴分离则不相交
#     """
#     # 提取两个OBB的主轴（旋转矩阵的列向量）
#     axes1 = [rot_mat1[:, 0], rot_mat1[:, 1], rot_mat1[:, 2]]
#     axes2 = [rot_mat2[:, 0], rot_mat2[:, 1], rot_mat2[:, 2]]
    
#     # 检查所有分离轴
#     all_axes = []
#     # 轴1：OBB1的主轴
#     all_axes.extend(axes1)
#     # 轴2：OBB2的主轴
#     all_axes.extend(axes2)
#     # 轴3：两两主轴的叉乘
#     for a1 in axes1:
#         for a2 in axes2:
#             cross_axis = np.cross(a1, a2)
#             if np.linalg.norm(cross_axis) > 1e-8:  # 避免零向量
#                 all_axes.append(cross_axis)
    
#     # 对每个轴检查投影是否重叠
#     for axis in all_axes:
#         min1, max1 = project_vertices_on_axis(vertices1, axis)
#         min2, max2 = project_vertices_on_axis(vertices2, axis)
#         # 投影无重叠 → 分离，返回False
#         if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
#             return False
#     return True


# def compute_convex_polytope_volume(vertices: np.ndarray) -> float:
#     """
#     计算凸多面体的体积（四面体分解法，以第一个顶点为公共顶点）
#     :param vertices: Nx3 凸多面体顶点数组（N≥4）
#     :return: 体积（非负）
#     """
#     if len(vertices) < 4:
#         return 0.0
#     volume = 0.0
#     v0 = vertices[0]
#     # 遍历所有四面体组合
#     for i in range(1, len(vertices)-2):
#         v1 = vertices[i]
#         v2 = vertices[i+1]
#         v3 = vertices[i+2]
#         # 向量
#         vec1 = v1 - v0
#         vec2 = v2 - v0
#         vec3 = v3 - v0
#         # 标量三重积 / 6 = 四面体体积
#         scalar_triple = np.dot(vec1, np.cross(vec2, vec3))
#         volume += abs(scalar_triple) / 6.0
#     return volume


# def clip_obb_to_obb(vertices1: np.ndarray, vertices2: np.ndarray) -> Optional[np.ndarray]:
#     """
#     裁剪两个OBB的相交区域，返回相交凸多面体的顶点（简化版，适用于OBB相交场景）
#     注：这是简化实现，工业级可使用更精准的库（如PyVista/Trimesh）
#     """
#     # 合并两个OBB的顶点并去重（初步筛选相交区域顶点）
#     all_vertices = np.vstack([vertices1, vertices2])
#     # 去重（浮点精度问题，用1e-6阈值）
#     unique_vertices = []
#     for v in all_vertices:
#         if not any(np.linalg.norm(v - uv) < 1e-6 for uv in unique_vertices):
#             unique_vertices.append(v)
#     unique_vertices = np.array(unique_vertices)
    
#     # 过滤出同时在两个OBB内部的顶点（简化判断：顶点到OBB各面的距离≥0）
#     # 此处为简化实现，若需更高精度可使用GJK/EPA算法
#     if len(unique_vertices) < 4:
#         return None
#     return unique_vertices


# def calculate_obb_volume(size: np.ndarray) -> float:
#     """计算单个OBB的体积（尺寸乘积）"""
#     return float(np.prod(size))


# def calculate_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
#     """
#     计算两个带旋转的3D有向边界框（OBB）的IoU
#     bbox格式：[cx, cy, cz, sx, sy, sz, rx, ry, rz]
#     - cx/cy/cz: 中心点坐标
#     - sx/sy/sz: 沿OBB主轴的尺寸（长度）
#     - rx/ry/rz: 欧拉角（绕x/y/z轴旋转，单位：弧度，ZYX顺序）
#     """
#     try:
#         # 解析OBB参数（前3=中心，3-6=尺寸，6-9=欧拉角）
#         if len(bbox1) < 9 or len(bbox2) < 9:
#             return 0.0
        
#         # 提取参数并转换为numpy数组
#         c1 = np.array(bbox1[:3], dtype=np.float64)
#         s1 = np.array(bbox1[3:6], dtype=np.float64)
#         r1 = np.array(bbox1[6:9], dtype=np.float64)
        
#         c2 = np.array(bbox2[:3], dtype=np.float64)
#         s2 = np.array(bbox2[3:6], dtype=np.float64)
#         r2 = np.array(bbox2[6:9], dtype=np.float64)
        
#         # 校验尺寸有效性（必须为正）
#         if np.any(s1 <= 1e-8) or np.any(s2 <= 1e-8):
#             return 0.0
        
#         # 1. 生成旋转矩阵
#         rot_mat1 = euler_to_rotation_matrix(r1)
#         rot_mat2 = euler_to_rotation_matrix(r2)
        
#         # 2. 生成两个OBB的顶点
#         vertices1 = get_obb_vertices(c1, s1, rot_mat1)
#         vertices2 = get_obb_vertices(c2, s2, rot_mat2)
        
#         # 3. SAT判断是否相交
#         if not sat_3d_obb_intersection(vertices1, rot_mat1, vertices2, rot_mat2):
#             # print("++++++++++++++++++++++++++++++")
#             # print("++++++++++++++++++++++++++++++")
#             # print("++++++++++++++++++++++++++++++")
#             # print("++++++++++++++++++++++++++++++")
#             return 0.0
        
#         # # 4. 计算相交体积（简化版裁剪）
#         # intersection_vertices = clip_obb_to_obb(vertices1, vertices2)
#         # if intersection_vertices is None or len(intersection_vertices) < 4:
#         #     inter_volume = 0.0
#         # else:
#         #     inter_volume = compute_convex_polytope_volume(intersection_vertices)
#         # 4. 精准计算相交体积（trimesh布尔交集运算）
#         # 布尔交集：计算两个mesh的重叠区域
#         intersection_mesh = trimesh.boolean.intersection([mesh1, mesh2], engine='blender')
#         # print("++++++++++++++++++++++++++++++")
#         # print("++++++++++++++++++++++++++++++")
#         # print("++++++++++++++++++++++++++++++")
#         # print("++++++++++++++++++++++++++++++")
#         print("intersection_mesh:",intersection_mesh)
#         if intersection_mesh is None or len(intersection_mesh.vertices) < 4:
#             inter_volume = 0.0
#         else:
#             # 计算相交mesh的体积（trimesh内置精准体积计算）
#             inter_volume = float(intersection_mesh.volume)
        
#         # 5. 计算两个OBB的体积
#         vol1 = calculate_obb_volume(s1)
#         vol2 = calculate_obb_volume(s2)
        
#         # 6. 计算IoU
#         union_volume = vol1 + vol2 - inter_volume
#         if union_volume <= 1e-8:
#             return 0.0
        
#         iou = inter_volume / union_volume
#         return float(max(0.0, min(iou, 1.0)))  # 确保IoU在0-1之间
    
#     except Exception as e:
#         # 捕获所有异常，返回0（鲁棒性）
#         return 0.0
# import numpy as np
# from scipy.spatial import ConvexHull
# from typing import List, Tuple

# # 必要常量定义（参考原逻辑）
# _POINT_IN_FRONT_OF_PLANE = 1
# _POINT_BEHIND_PLANE = -1
# _POINT_ON_PLANE = 0
# _PLANE_THICKNESS_EPSILON = 1e-6

# # 3D立方体的6个面，每个面包含4个顶点索引（与OrientedBBox的顶点生成顺序对应）
# FACES = np.array([
#     [0, 1, 2, 3],  # +Z面
#     [4, 5, 6, 7],  # -Z面
#     [0, 1, 5, 4],  # +Y面
#     [2, 3, 7, 6],  # -Y面
#     [0, 3, 7, 4],  # +X面
#     [1, 2, 6, 5],  # -X面
# ])
# NUM_KEYPOINTS = 8  # 3D包围盒的8个顶点


# class OrientedBBox:
#     """有向3D包围盒类，封装中心、尺寸、旋转及核心计算方法"""
#     def __init__(self, bbox: List[float]):
#         """
#         初始化有向3D包围盒
#         bbox格式：[cx, cy, cz, sx, sy, sz, rx, ry, rz]
#         - cx/cy/cz: 包围盒中心坐标
#         - sx/sy/sz: 包围盒在自身局部坐标系下的尺寸（轴对齐）
#         - rx/ry/rz: 绕X/Y/Z轴的旋转角（欧拉角，单位：弧度）
#         """
#         # 解析包围盒参数
#         self.center = np.array(bbox[:3], dtype=np.float64)  # 中心 (cx, cy, cz)
#         self.size = np.array(bbox[3:6], dtype=np.float64)   # 尺寸 (sx, sy, sz)
#         self.rotation_euler = np.array(bbox[6:9], dtype=np.float64) if len(bbox)>=9 else np.zeros(3)
        
#         # 计算旋转矩阵（欧拉角转旋转矩阵，Z-Y-X顺序）
#         self.rotation = self._euler_to_rotation(self.rotation_euler)
#         # 平移向量（即中心）
#         self.translation = self.center
#         # 变换矩阵（旋转+平移）
#         self.transformation = self._get_transformation_matrix()
#         # 生成包围盒的8个顶点（世界坐标系）
#         self.vertices = self._generate_vertices()
#         # 包围盒体积
#         self.volume = self._compute_volume()

#     # def _euler_to_rotation(self, euler: np.ndarray) -> np.ndarray:
#     #     """欧拉角转旋转矩阵（Z-Y-X顺序）"""
#     #     rx, ry, rz = euler
#     #     # 绕X轴旋转
#     #     R_x = np.array([
#     #         [1, 0, 0],
#     #         [0, np.cos(rx), -np.sin(rx)],
#     #         [0, np.sin(rx), np.cos(rx)]
#     #     ])
#     #     # 绕Y轴旋转
#     #     R_y = np.array([
#     #         [np.cos(ry), 0, np.sin(ry)],
#     #         [0, 1, 0],
#     #         [-np.sin(ry), 0, np.cos(ry)]
#     #     ])
#     #     # 绕Z轴旋转
#     #     R_z = np.array([
#     #         [np.cos(rz), -np.sin(rz), 0],
#     #         [np.sin(rz), np.cos(rz), 0],
#     #         [0, 0, 1]
#     #     ])
#     #     # 组合旋转矩阵: R = Rz * Ry * Rx
#     #     R = np.dot(R_z, np.dot(R_y, R_x))
#     #     return R
#     from scipy.spatial.transform import Rotation as R_scipy
#     def _euler_to_rotation(self, euler):
#         return R_scipy.from_euler("xyz", euler, degrees=False).as_matrix()

#     def _get_transformation_matrix(self) -> np.ndarray:
#         """生成4x4的变换矩阵（旋转+平移）"""
#         transform = np.eye(4, dtype=np.float64)
#         transform[:3, :3] = self.rotation
#         transform[:3, 3] = self.translation
#         return transform

#     def _generate_vertices(self) -> np.ndarray:
#         """生成包围盒的8个顶点（世界坐标系）"""
#         # 生成局部坐标系下的轴对齐立方体顶点（中心在原点）
#         half_size = self.size / 2.0
#         local_vertices = np.array([
#             [-half_size[0], -half_size[1], -half_size[2]],
#             [half_size[0], -half_size[1], -half_size[2]],
#             [half_size[0], half_size[1], -half_size[2]],
#             [-half_size[0], half_size[1], -half_size[2]],
#             [-half_size[0], -half_size[1], half_size[2]],
#             [half_size[0], -half_size[1], half_size[2]],
#             [half_size[0], half_size[1], half_size[2]],
#             [-half_size[0], half_size[1], half_size[2]],
#         ], dtype=np.float64)
#         # 应用旋转和平移转换到世界坐标系
#         world_vertices = np.dot(local_vertices, self.rotation.T) + self.center
#         return world_vertices

#     def _compute_volume(self) -> float:
#         """计算包围盒体积（有向包围盒体积与轴对齐时一致）"""
#         return float(np.prod(self.size))

#     def apply_transformation(self, transform: np.ndarray) -> 'OrientedBBox':
#         """将包围盒应用4x4变换矩阵，返回新的包围盒（仅用于坐标变换）"""
#         # 转换顶点到新坐标系
#         hom_vertices = np.hstack([self.vertices, np.ones((8, 1))])  # 齐次坐标
#         transformed_vertices = np.dot(hom_vertices, transform.T)[:, :3]
        
#         # 构建临时包围盒（仅保留顶点和基础属性，用于交点计算）
#         temp_box = OrientedBBox([0,0,0,1,1,1,0,0,0])
#         temp_box.vertices = transformed_vertices
#         temp_box.center = np.mean(transformed_vertices, axis=0)
#         temp_box.rotation = np.eye(3)  # 变换后可能是轴对齐，暂设为单位矩阵
#         temp_box.translation = temp_box.center
#         return temp_box

#     def inside(self, point: np.ndarray) -> bool:
#         """判断点是否在包围盒内（转换到局部坐标系后判断轴对齐）"""
#         # 将点转换到包围盒的局部坐标系
#         local_point = np.dot(point - self.center, self.rotation)
#         # 局部坐标系下是轴对齐，判断是否在范围内
#         half_size = self.size / 2.0
#         return np.all(np.abs(local_point) <= half_size + _PLANE_THICKNESS_EPSILON)


# def calculate_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
#     """
#     计算两个3D有向包围盒的IoU（支持轴对齐包围盒，此时旋转角为0）
    
#     Args:
#         bbox1: 第一个包围盒，格式 [cx, cy, cz, sx, sy, sz, rx, ry, rz]
#         bbox2: 第二个包围盒，格式 [cx, cy, cz, sx, sy, sz, rx, ry, rz]
    
#     Returns:
#         float: IoU值（0~1）
#     """
#     try:
#         # 初始化有向包围盒对象
#         box1 = OrientedBBox(bbox1)
#         box2 = OrientedBBox(bbox2)
#     except Exception as e:
#         return 0.0

#     # 存储所有交点
#     intersection_points = []

#     def _classify_point_to_plane(point: np.ndarray, plane: np.ndarray, normal: float, axis: int) -> int:
#         """分类点相对于平面的位置（前/后/在平面上）"""
#         signed_distance = normal * (point[axis] - plane[axis])
#         if signed_distance > _PLANE_THICKNESS_EPSILON:
#             return _POINT_IN_FRONT_OF_PLANE
#         elif signed_distance < -_PLANE_THICKNESS_EPSILON:
#             return _POINT_BEHIND_PLANE
#         else:
#             return _POINT_ON_PLANE

#     # def _intersect(plane: np.ndarray, prev_point: np.ndarray, current_point: np.ndarray, axis: int) -> np.ndarray:
#     #     """计算线段与轴对齐平面的交点"""
#     #     denominator = current_point[axis] - prev_point[axis]
#     #     if np.abs(denominator) < 1e-9:
#     #         return prev_point  # 线段平行于平面，返回任意点（实际不会走到这里）
#     #     alpha = (current_point[axis] - plane[axis]) / denominator
#     #     intersection_point = alpha * prev_point + (1.0 - alpha) * current_point
#     #     print(11111111)
#     #     return intersection_point
#     def _intersect(plane: np.ndarray, prev_point: np.ndarray, current_point: np.ndarray, axis: int) -> np.ndarray:
#         """计算线段与轴对齐平面的交点（修正版）"""
#         denom = current_point[axis] - prev_point[axis]
#         if abs(denom) < 1e-12:
#             return prev_point  # 线段平行于平面
#         t = (plane[axis] - prev_point[axis]) / denom
#         return prev_point + t * (current_point - prev_point)


#     def _clip_poly(poly: List[np.ndarray], plane: np.ndarray, normal: float, axis: int) -> List[np.ndarray]:
#         """Sutherland-Hodgman算法：用平面裁剪多边形"""
#         result = []
#         if len(poly) <= 1:
#             return result

#         poly_in_plane = True  # 标记多边形是否完全在平面上

#         for i, curr_p in enumerate(poly):
#             prev_p = poly[(i + len(poly) - 1) % len(poly)]
#             d1 = _classify_point_to_plane(prev_p, plane, normal, axis)
#             d2 = _classify_point_to_plane(curr_p, plane, normal, axis)

#             if d2 == _POINT_BEHIND_PLANE:
#                 poly_in_plane = False
#                 if d1 == _POINT_IN_FRONT_OF_PLANE:
#                     # 前点->后点，计算交点并加入
#                     intersect_p = _intersect(plane, prev_p, curr_p, axis)
#                     result.append(intersect_p)
#                 elif d1 == _POINT_ON_PLANE:
#                     # 前点在平面上，加入前点（避免重复）
#                     if not result or not np.allclose(result[-1], prev_p, atol=1e-6):
#                         result.append(prev_p)
#             elif d2 == _POINT_IN_FRONT_OF_PLANE:
#                 poly_in_plane = False
#                 if d1 == _POINT_BEHIND_PLANE:
#                     # 后点->前点，计算交点并加入
#                     intersect_p = _intersect(plane, prev_p, curr_p, axis)
#                     result.append(intersect_p)
#                 elif d1 == _POINT_ON_PLANE:
#                     # 前点在平面上，加入前点（避免重复）
#                     if not result or not np.allclose(result[-1], prev_p, atol=1e-6):
#                         result.append(prev_p)
#                 # 加入当前前点
#                 result.append(curr_p)
#             else:
#                 # 当前点在平面上，且前点不在平面上时加入
#                 if d1 != _POINT_ON_PLANE:
#                     result.append(curr_p)

#         return poly if poly_in_plane else result

#     def intersect_box_poly(box: OrientedBBox, poly: List[np.ndarray]) -> List[np.ndarray]:
#         """用轴对齐包围盒的6个面裁剪多边形"""
#         # 轴对齐包围盒的极值点（用于定义裁剪平面）
#         min_p = np.min(box.vertices, axis=0)
#         max_p = np.max(box.vertices, axis=0)
        
#         for axis in range(3):
#             # 裁剪+axis方向的平面（max_p）
#             poly = _clip_poly(poly, max_p, 1.0, axis)
#             # 裁剪-axis方向的平面（min_p）
#             poly = _clip_poly(poly, min_p, -1.0, axis)
#         return poly

#     def compute_intersection_points(box_src: OrientedBBox, box_template: OrientedBBox):
#         """计算两个包围盒的交点"""
#         # 将源包围盒转换为轴对齐（求逆变换）
#         inv_transform = np.linalg.inv(box_src.transformation)
#         box_src_aligned = box_src.apply_transformation(inv_transform)
#         template_in_src = box_template.apply_transformation(inv_transform)

#         # 1. 裁剪模板包围盒的每个面，计算与源包围盒的交点
#         for face_idx in range(len(FACES)):
#             face_vertices_idx = FACES[face_idx]
#             # 取当前面的4个顶点
#             poly = [template_in_src.vertices[idx] for idx in face_vertices_idx]
#             # 裁剪多边形，得到交点（源坐标系下）
#             clip_points = intersect_box_poly(box_src_aligned, poly)
           
#             for p in clip_points:
#                 p_world = np.dot(box_src.rotation, p) + box_src.translation
#                 intersection_points.append(p_world)

#         # 2. 检查模板包围盒的顶点是否在源包围盒内
#         for p in template_in_src.vertices:
#             if box_src_aligned.inside(p):
#                 p_world = np.dot(box_src.rotation, p) + box_src.translation
#                 intersection_points.append(p_world)

#     # 计算双向交点（box1→box2 和 box2→box1）
#     compute_intersection_points(box1, box2)
#     compute_intersection_points(box2, box1)

#     # 去重交点（避免重复计算）
#     if intersection_points:
#         intersection_points = np.array(intersection_points)
#         # 去重（浮点精度问题，用tol判断）
#         unique_mask = np.ones(len(intersection_points), dtype=bool)
#         for i in range(len(intersection_points)):
#             if unique_mask[i]:
#                 # 标记后续重复点
#                 duplicate = np.allclose(intersection_points[i], intersection_points[i+1:], atol=1e-6)
#                 unique_mask[i+1:][duplicate] = False
#         intersection_points = intersection_points[unique_mask]

#     # # 计算IoU
#     # if len(intersection_points) >= 4:  # 至少4个点才能构成3D凸包
#     #     try:
#     #         # 构建凸包并计算交集体积
#     #         hull = ConvexHull(intersection_points)
#     #         intersection_volume = hull.volume
#     #     except Exception:
#     #         # 凸包构建失败（点共面/共线），交集体积为0
#     #         intersection_volume = 0.0
#     # else:
#     #     intersection_volume = 0.0
#     if len(intersection_points) >= 4:
#         try:
#             hull = ConvexHull(intersection_points)
#             # 检查是否为 3D（凸包的 dimension == 3）
#             if hull.vertices.shape[0] < 4 or hull.area == 0:
#                 intersection_volume = 0.0
#             else:
#                 # 更鲁棒：检查点是否共面
#                 # 计算点云的协方差矩阵的最小特征值
#                 centroid = np.mean(intersection_points, axis=0)
#                 _, s, _ = np.linalg.svd(intersection_points - centroid)
#                 # 如果最小奇异值 < 阈值 → 共面
#                 if s[2] < 1e-6:  # 阈值可调
#                     intersection_volume = 0.0
#                 else:
#                     intersection_volume = hull.volume
#         except:
#             intersection_volume = 0.0
#     else:
#         intersection_volume = 0.0

#     # 计算并集体积
#     box1_volume = box1.volume
#     box2_volume = box2.volume
#     union_volume = box1_volume + box2_volume - intersection_volume
#     iou = intersection_volume / union_volume if union_volume > 1e-9 else 0.0
#     if iou > 1:
#         return 0.0
#     if union_volume <= 1e-9:
#         return 0.0
#     print("union_volume:", union_volume)
#     return float(iou)

# import numpy as np
# from scipy.spatial import ConvexHull
# from scipy.spatial.transform import Rotation as R
# from typing import List

# def calculate_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
#     """
#     计算两个3D有向包围盒的IoU。
    
#     Args:
#         bbox1: [cx, cy, cz, sx, sy, sz, rx, ry, rz] (欧拉角单位为弧度，旋转顺序假设为 xyz)
#         bbox2: [cx, cy, cz, sx, sy, sz, rx, ry, rz]
    
#     Returns:
#         float: IoU值 (0~1)
#     """
#     # 1. 计算两个Box的体积 (sx * sy * sz)
#     vol1 = bbox1[3] * bbox1[4] * bbox1[5]
#     vol2 = bbox2[3] * bbox2[4] * bbox2[5]
    
#     # 2. 获取两个Box的世界坐标系角点和旋转矩阵
#     corners1, rot1, center1, size1 = get_box_info(bbox1)
#     corners2, rot2, center2, size2 = get_box_info(bbox2)
    
#     # 3. 收集交集多面体的潜在顶点
#     # 3a. 包含在对方内部的角点
#     inter_points = []
    
#     # Box1 corners inside Box2
#     corners1_in_2 = world_to_local(corners1, center2, rot2)
#     mask1 = np.all(np.abs(corners1_in_2) <= size2 / 2 + 1e-6, axis=1)
#     inter_points.extend(corners1[mask1])
    
#     # Box2 corners inside Box1
#     corners2_in_1 = world_to_local(corners2, center1, rot1)
#     mask2 = np.all(np.abs(corners2_in_1) <= size1 / 2 + 1e-6, axis=1)
#     inter_points.extend(corners2[mask2])
    
#     # 3b. 棱与面的交点
#     # 检查 Box1 的棱 与 Box2 的面 的交点
#     edges1 = get_edges(corners1)
#     for p1, p2 in edges1:
#         # 将棱转换到 Box2 的局部坐标系
#         p1_local = world_to_local(p1.reshape(1, 3), center2, rot2)[0]
#         p2_local = world_to_local(p2.reshape(1, 3), center2, rot2)[0]
#         pts = line_box_intersection(p1_local, p2_local, size2)
#         if len(pts) > 0:
#             # 转回世界坐标
#             pts_world = local_to_world(np.array(pts), center2, rot2)
#             inter_points.extend(pts_world)
            
#     # 检查 Box2 的棱 与 Box1 的面 的交点
#     edges2 = get_edges(corners2)
#     for p1, p2 in edges2:
#         # 将棱转换到 Box1 的局部坐标系
#         p1_local = world_to_local(p1.reshape(1, 3), center1, rot1)[0]
#         p2_local = world_to_local(p2.reshape(1, 3), center1, rot1)[0]
#         pts = line_box_intersection(p1_local, p2_local, size1)
#         if len(pts) > 0:
#             pts_world = local_to_world(np.array(pts), center1, rot1)
#             inter_points.extend(pts_world)
            
#     inter_points = np.array(inter_points)
    
#     if len(inter_points) < 4:
#         return 0.0
    
#     # 4. 使用 ConvexHull 计算交集体积
#     try:
#         hull = ConvexHull(inter_points)
#         inter_vol = hull.volume
#     except Exception:
#         # 如果点共面或少于4个，ConvexHull会报错，说明体积为0
#         return 0.0
    
#     # 5. 计算 IoU
#     union_vol = vol1 + vol2 - inter_vol
#     if union_vol <= 0:
#         return 0.0
        
#     return inter_vol / union_vol

# # --- 辅助函数 ---

# def get_box_info(bbox):
#     cx, cy, cz, sx, sy, sz, rx, ry, rz = bbox
#     center = np.array([cx, cy, cz])
#     size = np.array([sx, sy, sz])
    
#     # 这里的旋转顺序 'xyz' 对应输入的 rx, ry, rz。
#     r = R.from_euler('xyz', [rx, ry, rz], degrees=False)
#     rot_matrix = r.as_matrix()
    
#     # 标准立方体的8个顶点 (未缩放)
#     template = np.array([
#         [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
#         [-0.5, -0.5, 0.5],  [0.5, -0.5, 0.5],  [0.5, 0.5, 0.5],  [-0.5, 0.5, 0.5]
#     ])
    
#     # 缩放 -> 旋转 -> 平移
#     corners = (template * size) @ rot_matrix.T + center
#     return corners, rot_matrix, center, size

# def world_to_local(points, center, rot_matrix):
#     """将世界坐标点转换到Box的局部坐标系（Box中心为原点，轴对齐）"""
#     return (points - center) @ rot_matrix

# def local_to_world(points, center, rot_matrix):
#     """局部坐标转世界坐标"""
#     return points @ rot_matrix.T + center

# def get_edges(corners):
#     """根据8个角点返回12条棱线段 (start_point, end_point)"""
#     # 顶点索引顺序基于上面的 template
#     idx_pairs = [
#         (0,1), (1,2), (2,3), (3,0), # 下底面
#         (4,5), (5,6), (6,7), (7,4), # 上底面
#         (0,4), (1,5), (2,6), (3,7)  # 侧棱
#     ]
#     return [(corners[i], corners[j]) for i, j in idx_pairs]

# def line_box_intersection(p1, p2, size):
#     """
#     计算线段 p1-p2 与 位于原点的轴对齐Box (尺寸size) 的交点。
#     返回线段在Box内部的部分的交点列表。
#     """
#     # 简单的 Liang-Barsky 算法变体或直接面相交检测
#     # 这里我们简化逻辑：线段与6个平面求交，看交点是否在面上
    
#     intersections = []
#     half_size = size / 2.0
    
#     # 检查两个端点是否已经在内部
#     if np.all(np.abs(p1) <= half_size + 1e-6): intersections.append(p1)
#     if np.all(np.abs(p2) <= half_size + 1e-6): intersections.append(p2)
    
#     # 如果两个点都在内部，直接返回
#     if len(intersections) == 2:
#         return intersections

#     # 定义6个面: 轴 (0,1,2) 及 方向 (+1, -1)
#     dir_vec = p2 - p1
    
#     for axis in range(3):
#         for sign in [-1, 1]:
#             plane_val = sign * half_size[axis]
            
#             # 避免除以0
#             if abs(dir_vec[axis]) < 1e-9:
#                 continue
                
#             # 计算参数 t: p1[axis] + t * dir[axis] = plane_val
#             t = (plane_val - p1[axis]) / dir_vec[axis]
            
#             # t 必须在 [0, 1] 之间才算在线段上
#             if 0 <= t <= 1:
#                 point = p1 + t * dir_vec
                
#                 # 检查交点是否在面的边界内 (其他两个维度)
#                 is_valid = True
#                 for other_axis in range(3):
#                     if other_axis == axis: continue
#                     if abs(point[other_axis]) > half_size[other_axis] + 1e-6:
#                         is_valid = False
#                         break
                
#                 if is_valid:
#                     # 去重添加
#                     if not any(np.allclose(point, x) for x in intersections):
#                         intersections.append(point)
                        
#     return intersections

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import List
# 导入指定的 API
from cutoop.iou import compute_3d_bbox_iou

def calculate_iou_3d(bbox1: List[float], bbox2: List[float]) -> float:
    """
    使用 cutoop.iou.compute_3d_bbox_iou 计算 3D IoU。
    
    Args:
        bbox1: [cx, cy, cz, sx, sy, sz, rx, ry, rz]
        bbox2: [cx, cy, cz, sx, sy, sz, rx, ry, rz]
        
    Returns:
        float: IoU 值
    """
    # 1. 解析 bbox1
    tA = np.array(bbox1[0:3])       # Translation (cx, cy, cz)
    sA = np.array(bbox1[3:6])       # Size (sx, sy, sz)
    rA_angles = bbox1[6:9]          # Euler angles
    # 将欧拉角转为 3x3 旋转矩阵 (保持之前的 'xyz' 顺序)
    rA = R.from_euler('xyz', rA_angles, degrees=False).as_matrix()
    # 2. 解析 bbox2
    tB = np.array(bbox2[0:3])
    sB = np.array(bbox2[3:6])
    rB_angles = bbox2[6:9]
    rB = R.from_euler('xyz', rB_angles, degrees=False).as_matrix()

    # 3. 调用 API
    iou = compute_3d_bbox_iou(
        rA_3x3=rA, 
        tA_3=tA, 
        sA_3=sA, 
        rB_3x3=rB, 
        tB_3=tB, 
        sB_3=sB
    )

    print("iou:", iou)
    return iou


# def oriented_bbox_2d_metric(gt: Any, pred: Any) -> Optional[Dict[str, Any]]:
#     """Compute metrics for 2d oriented bboxes as described:

#     - For the first two items (indexes 0 and 1): compute per-dimension absolute differences and L2 norm
#     - For the middle two items (indexes 2 and 3): compute IoU loss (1 - IoU)
#     - For the last item (index 4): compute rotation angle absolute difference (in radians normalized)

#     Returns a list of per-object dicts with keys: 'l2_xy', 'iou', 'angle_diff'
#     """
#     g = _extract_oriented_bboxes_from(gt)
#     p = _extract_oriented_bboxes_from(pred)
#     print("g:", g)
#     print("p:", p)
#     if not g or not p:
#         return None

#     # helper to normalize angle diff to [-pi, pi]
#     def _angdiff(a: float, b: float) -> float:
#         d = (a - b + math.pi) % (2 * math.pi) - math.pi
#         return abs(d)

#     per_object: List[Dict[str, Optional[float]]] = []
#     n = min(len(g), len(p))
#     for k in range(n):
#         gb = g[k][1]
#         pb = p[k][1]
#         # ensure both are lists of at least length 5
#         if not (isinstance(gb, (list, tuple)) and isinstance(pb, (list, tuple)) and len(gb) >= 5 and len(pb) >= 5):
#             per_object.append({"l2_xy": None, "iou": None, "angle_diff": None})
#             continue

#         # L2 on first two dimensions (cx, cy)
#         try:
#             l2_xy = _euclidean([float(gb[0]), float(gb[1])], [float(pb[0]), float(pb[1])])
#         except Exception:
#             l2_xy = None

#         # IoU on full bbox (uses Shapely-backed implementation when available)
#         try:
#             print("________________1-",list(map(float, gb[:5])))
#             print("________________2-",list(map(float, pb[:5])))
          
#             iou = calculate_iou_2d_oriented(list(map(float, gb[:5])), list(map(float, pb[:5])))
#             print("iou:", iou)
#         except Exception:
#             iou = None

#         # angle diff (abs normalized)
#         try:
#             angle_d = _angdiff(float(gb[4]), float(pb[4]))
#         except Exception:
#             angle_d = None

#         # per_object.append({"l2_xy": l2_xy, "iou": iou, "angle_diff": angle_d, "metrics"})
#         # per_object.append(iou+angle_d/(2*3.1416))
#         per_object = iou

#     return per_object
import math
import numpy as np
from shapely.geometry import Polygon

def oriented_bbox_2d_metric(gt: Any, pred: Any) -> Optional[float]:
    """
    仅返回单对旋转框的 IoU 值 (float)。
    """
    # 1. 提取数据
    g_extracted = _extract_oriented_bboxes_from(gt)
    p_extracted = _extract_oriented_bboxes_from(pred)

    if not g_extracted or not p_extracted:
        return 0.0

    # 2. 获取参数 [cx, cy, w, h, theta]
    # 根据你的日志，数据位于提取结果的 [0][1]
    gb = g_extracted[0][1]
    pb = p_extracted[0][1]

    def get_polygon(bbox):
        cx, cy, w, h, theta = bbox
        dx, dy = w / 2.0, h / 2.0
        # 定义四个顶点坐标
        corners = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
        # 旋转矩阵
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        # 旋转并平移
        rotated_corners = corners @ R.T + np.array([cx, cy])
        return Polygon(rotated_corners)

    try:
        poly_g = get_polygon(gb[:5])
        poly_p = get_polygon(pb[:5])

        if not poly_g.intersects(poly_p):
            return 0.0

        inter_area = poly_g.intersection(poly_p).area
        union_area = poly_g.area + poly_p.area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    except Exception:
        return 0.0
        
def obj_bbox_qa_detect(gt: Any, pred: Any) -> Optional[float]:
    """Compute mean 3D IoU for detected object bboxes between GT and prediction.

    Behavior:
    - Extract lists of (label, bbox3d) from GT and pred using `_extract_bbox3d_from`.
    - If labels present for both sides, align by normalized label (case-insensitive);
      otherwise pair by index.
    - Compute 3D IoU per matched pair using calculate_iou_3d and return mean IoU (float).
    - Return None if no valid pairs found.
    """
    g = _extract_bbox3d_from(gt)
    p = _extract_bbox3d_from(pred)
    logger.info(str(g))
    logger.info(str(p))
    if not g or not p:
        return None

    # Build label maps (normalized)
    g_map = {(_normalize_label(lbl) if lbl else None): bbox for lbl, bbox in g}
    p_map = {(_normalize_label(lbl) if lbl else None): bbox for lbl, bbox in p}

    vals: List[float] = []
    # If labels exist on both sides, align by label
    # if any(k is not None for k in g_map.keys()) and any(k is not None for k in p_map.keys()):
    #     for lbl, gbbox in g_map.items():
    #         if lbl is not None and lbl in p_map:
    #             vals.append(calculate_iou_3d(gbbox, p_map[lbl]))
    # else:
    # pair by index
    n = min(len(g), len(p))
    for i in range(n):
        logger.info(str(g[i][1]))
        logger.info(str(p[i][1]))
        vals.append(calculate_iou_3d(g[i][1], p[i][1]))
        #print("vals:", vals)

    if not vals:
        return None
    return float(sum(vals) / len(vals))


# ======================
# Point Metrics
# ======================
def _extract_points_from(obj: Any) -> List[Tuple[Optional[str], List[float]]]:
    """Return list of (label, point) pairs extracted from obj.

    Supports:
    - a single point list [x,y] or [x,y,z] -> returns [(None, point)]
    - a list of dicts where each dict may contain keys like 'point_2d','point_3d','functional_point_2d','functional_point_3d','grasp_point_2d'
      and optional 'label'
    - a dict with these keys -> returns single item
    """
    out: List[Tuple[Optional[str], List[float]]] = []
    if obj is None:
        return out

    # Single numeric list
    if _is_number_list(obj) and len(obj) in (2, 3):
        return [(None, list(obj))]

    # Dict with embedded point
    if isinstance(obj, dict):
        # try common keys
        for k in ("functional_point_3d", "functional_point_2d", "point_3d", "point_2d", "grasp_point_3d", "grasp_point_2d"):
            if k in obj and _is_number_list(obj[k]):
                label = obj.get("label") or obj.get("name")
                return [(label, list(obj[k]))]
        # If dict is label->point mapping
        # fallback: scan values for a numeric list
        for kk, vv in obj.items():
            if _is_number_list(vv) and len(vv) in (2, 3):
                return [(None, list(vv))]

    # List of items
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if item is None:
                continue
            if _is_number_list(item) and len(item) in (2, 3):
                out.append((None, list(item)))
                continue
            if isinstance(item, dict):
                # try to find point in dict
                label = item.get("label") or item.get("name")
                found = None
                for k in ("functional_point_3d", "functional_point_2d", "point_3d", "point_2d", "grasp_point_3d", "grasp_point_2d"):
                    if k in item and (_is_number_list(item[k]) or _is_number_list(item[k][0])):
                        found = list(item[k])
                        break
                if found is None:
                    # search any numeric list value
                    for vv in item.values():
                        if _is_number_list(vv) and len(vv) in (2, 3):
                            found = list(vv)
                            break
                if found is not None:
                    out.append((label, found))
        return out

    return out

def l2_loss_single(gt: Any, pred: Any) -> Optional[float]:
    """Compute L2 distance between single GT point and pred point."""
    g = _extract_points_from(gt)
    p = _extract_points_from(pred)
    if not g or not p:
        return None

    # Single logic: take the first zip-paired element and handle nested structures
    for gi, pi in zip(g, p):
        # gi and pi are (label, point)
        gi_c = gi[1] if not isinstance(gi[1][0], list) else gi[1][0]
        pi_c = pi[1] if not isinstance(pi[1][0], list) else pi[1][0]
        return _euclidean(gi_c, pi_c)

    return None

def mean_l2_loss(gt: Any, pred: Any) -> Optional[float]:
    print(gt, pred)
    g = _extract_points_from(gt)
    p = _extract_points_from(pred)
    print(g, p)
    if not g or not p:
        return None

    # Default: assume element-wise alignment and pair by zip.
    vals: List[float] = []
    for gi, pi in zip(g, p):
        # gi and pi are (label, point)
        gi_c = gi[1] if not isinstance(gi[1][0], list) else gi[1][0]
        pi_c = pi[1] if not isinstance(pi[1][0], list) else pi[1][0]
        vals.append(_euclidean(gi_c, pi_c))

    if not vals:
        return None
    return float(sum(vals) / len(vals))


# ======================
# Trajectory Metrics
# ======================
def _discrete_frechet_distance(traj1: List[List[float]], traj2: List[List[float]]) -> float:
    if len(traj1) == 0 or len(traj2) == 0:
        return float('inf')
    
    n, m = len(traj1), len(traj2)
    ca = np.full((n, m), -1.0)
    
    def euclidean_dist(p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
    
    def compute(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = euclidean_dist(traj1[i], traj2[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i > 0 and j == 0:
            ca[i, j] = max(compute(i-1, 0), d)
        elif i == 0 and j > 0:
            ca[i, j] = max(compute(0, j-1), d)
        elif i > 0 and j > 0:
            ca[i, j] = max(min(compute(i-1, j), compute(i-1, j-1), compute(i, j-1)), d)
        else:
            ca[i, j] = d
        return ca[i, j]
    
    return compute(n-1, m-1)

def _hausdorff_distance(traj1: List[List[float]], traj2: List[List[float]]) -> float:
    if len(traj1) == 0 or len(traj2) == 0:
        return float('inf')
    
    def directed_hausdorff_distance(set1, set2):
        max_dist = 0
        for p1 in set1:
            min_dist = float('inf')
            for p2 in set2:
                dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
                min_dist = min(min_dist, dist)
            max_dist = max(max_dist, min_dist)
        return max_dist
    
    return max(directed_hausdorff_distance(traj1, traj2), 
               directed_hausdorff_distance(traj2, traj1))

def _rmse_distance(traj1: List[List[float]], traj2: List[List[float]]) -> float:
    if len(traj1) == 0 or len(traj2) == 0:
        return float('inf')
    min_len = min(len(traj1), len(traj2))
    if min_len == 0:
        return float('inf')
    squared_errors = []
    for i in range(min_len):
        p1, p2 = traj1[i], traj2[i]
        dist_squared = sum((a - b) ** 2 for a, b in zip(p1, p2))
        squared_errors.append(dist_squared)
    return math.sqrt(sum(squared_errors) / len(squared_errors))

# def evaluate_3d_trajectory(gt: Any, pred: Any) -> Optional[Dict[str, float]]:
#     """Evaluate 3D trajectory similarity between GT and prediction.

#     Returns dict with keys: 'Discrete_Fréchet_Distance', 'Hausdorff_Distance', 'RMSE'
#     or None if no valid pairs found.
#     """
#     """Evaluate a single 3D trajectory pair between GT and prediction.

#     This function assumes the caller provides a single trajectory for gt and a
#     single trajectory for pred (no id mapping). It tries several common
#     shapes:
#     - a plain list of 3D points: [[x,y,z], ...]
#     - a dict with key '3d_trajectory' or 'trajectory_3d' whose value is a list
#     - a JSON string that decodes to one of the above

#     Returns a dict with the three metrics, or None if unable to extract both
#     trajectories.
#     """

#     # Minimal extraction: expect gt and pred to contain a direct trajectory list
#     def _simple_3d_extract(obj: Any) -> List[List[float]]:
#         if obj is None:
#             return []
#         # dict with explicit key
#         if isinstance(obj, dict) and "3d_trajectory" in obj:
#             t = obj["3d_trajectory"]
#             if isinstance(t, (list, tuple)) and t and all(_is_number_list(pt) for pt in t):
#                 return [list(map(float, pt)) for pt in t]
#             return []
#         # direct list of points
#         if isinstance(obj, (list, tuple)) and obj and all(_is_number_list(pt) for pt in obj):
#             return [list(map(float, pt)) for pt in obj]
#         return []

#     gt_traj = _simple_3d_extract(gt[0])
#     pred_traj = _simple_3d_extract(pred[0])

#     logger.info(str(gt))
#     logger.info(str(pred))
#     logger.info(str(gt_traj))
#     logger.info(str(pred_traj))

#     if not gt_traj or not pred_traj:
#         return None

#     dfd = _discrete_frechet_distance(gt_traj, pred_traj)
#     hd = _hausdorff_distance(gt_traj, pred_traj)
#     rmse = _rmse_distance(gt_traj, pred_traj)

#     if math.isinf(dfd) or math.isinf(hd) or math.isinf(rmse):
#         return None

#     return {
#         'Discrete_Fréchet_Distance': float(dfd),
#         'Hausdorff_Distance': float(hd),
#         'RMSE': float(rmse)
#     }

# def evaluate_2d_trajectory(gt: Any, pred: Any) -> Optional[Dict[str, float]]:
#     """Evaluate 2D trajectory similarity between GT and prediction.

#     The function extracts '2d_trajectory' from the first returned item. If the
#     field is a dict of camera->list, it prefers 'Left_Camera' or the first
#     available camera.
#     """
#     """Evaluate a single 2D trajectory pair between GT and prediction.

#     Similar to the 3D version, this function expects one GT trajectory and one
#     predicted trajectory. It supports direct lists of 2D points or dicts with
#     key '2d_trajectory' whose value may be a camera->list mapping. If a
#     camera-mapped dict is provided it prefers 'Left_Camera' or the first
#     available camera.
#     """

#     # Minimal extraction: expect gt and pred to contain a direct 2D trajectory list
#     def _simple_2d_extract(obj: Any) -> List[List[float]]:
#         if obj is None:
#             return []
#         # dict with explicit key
#         if isinstance(obj, dict) and "2d_trajectory" in obj:
#             t = obj["2d_trajectory"]
#             if isinstance(t, (list, tuple)) and t and all(_is_number_list(pt) for pt in t):
#                 return [list(map(float, pt)) for pt in t]
#             return []
#         # direct list of points
#         if isinstance(obj, (list, tuple)) and obj and all(_is_number_list(pt) for pt in obj):
#             return [list(map(float, pt)) for pt in obj]
#         return []

#     gt_traj = _simple_2d_extract(gt[0])
#     pred_traj = _simple_2d_extract(pred[0])
#     print("gt[0]:", gt[0])
#     print("pred:", pred)
#     # logger.info(str(gt))
#     # logger.info(str(pred))
#     # logger.info(str(gt_traj))
#     # logger.info(str(pred_traj))
#     print("gt_traj:", gt_traj)
#     print("pred_traj:", pred_traj)

#     if not gt_traj or not pred_traj:
#         return None

#     dfd = _discrete_frechet_distance(gt_traj, pred_traj)
#     hd = _hausdorff_distance(gt_traj, pred_traj)
#     rmse = _rmse_distance(gt_traj, pred_traj)
#     print("=============================")
#     print("dfd:", dfd)
#     print("hd:", hd)
#     print("rmse:", rmse)
    
#     if math.isinf(dfd) or math.isinf(hd) or math.isinf(rmse):
#         return None

#     return {
#         'Discrete_Fréchet_Distance': float(dfd),
#         'Hausdorff_Distance': float(hd),
#         'RMSE': float(rmse)
#      }


import math
from typing import List, Any, Dict, Optional

def _extract_trajectory_safe(obj: Any, traj_key: str) -> (List[List[float]], str):
    """
    尝试从输入中安全提取轨迹。
    返回: (轨迹列表, 错误/原因说明)
    """
    # 1. 处理输入为列表的情况（如 gt: [{...}]）
    actual_obj = obj
    if isinstance(obj, list) or isinstance(obj, tuple):
        if len(obj) > 0:
            actual_obj = obj[0]
        else:
            return [], "输入列表为空 []"

    if actual_obj is None:
        return [], "数据对象为 None"

    # 2. 确定数据源
    traj_data = []
    reason = ""
    
    if isinstance(actual_obj, dict):
        if traj_key in actual_obj:
            traj_data = actual_obj[traj_key]
        elif "trajectory_3d" in actual_obj: # 兼容其他可能键名
            traj_data = actual_obj["trajectory_3d"]
        else:
            return [], f"字典中缺失键: '{traj_key}'。现有键: {list(actual_obj.keys())}"
    elif isinstance(actual_obj, (list, tuple)):
        traj_data = actual_obj
    else:
        return [], f"无法识别的数据格式: {type(actual_obj)}"

    # 3. 验证轨迹数据内容
    if not isinstance(traj_data, (list, tuple)) or len(traj_data) == 0:
        return [], "轨迹数据为空或不是列表格式"

    # 4. 转换点格式并简单验证
    try:
        cleaned_traj = []
        for pt in traj_data:
            if isinstance(pt, (list, tuple)):
                cleaned_traj.append([float(x) for x in pt])
            else:
                return [], f"轨迹点格式错误: {pt} 应为列表/元组"
        return cleaned_traj, "Success"
    except Exception as e:
        return [], f"数值转换失败: {str(e)}"

# ======================
# 3D 轨迹评估
# ======================
def evaluate_3d_trajectory(gt: Any, pred: Any) -> Optional[Dict[str, float]]:
    gt_traj, gt_reason = _extract_trajectory_safe(gt, "3d_trajectory")
    pred_traj, pred_reason = _extract_trajectory_safe(pred, "3d_trajectory")

    if not gt_traj or not pred_traj:
        print(f"[3D Eval] 返回 None 原因 -> GT: {gt_reason} | Pred: {pred_reason}")
        return None

    dfd = _discrete_frechet_distance(gt_traj, pred_traj)
    hd = _hausdorff_distance(gt_traj, pred_traj)
    rmse = _rmse_distance(gt_traj, pred_traj)

    if math.isinf(dfd) or math.isinf(hd) or math.isinf(rmse):
        print(f"[3D Eval] 返回 None 原因 -> 计算结果出现 inf")
        return None

    return {
        'Discrete_Fréchet_Distance': float(dfd),
        'Hausdorff_Distance': float(hd),
        'RMSE': float(rmse)
    }

# ======================
# 2D 轨迹评估
# ======================
def evaluate_2d_trajectory(gt: Any, pred: Any) -> Optional[Dict[str, float]]:
    gt_traj, gt_reason = _extract_trajectory_safe(gt, "2d_trajectory")
    pred_traj, pred_reason = _extract_trajectory_safe(pred, "2d_trajectory")

    if not gt_traj or not pred_traj:
        print(f"[2D Eval] 返回 None 原因 -> GT: {gt_reason} | Pred: {pred_reason}")
        # 针对你给出的输出进行特定提示
        if "2d_trajectory" in str(pred) and not pred_traj:
            print(f"提示: 请检查 Pred 数据点是否嵌套, 当前收到: {pred}")
        return None

    dfd = _discrete_frechet_distance(gt_traj, pred_traj)
    hd = _hausdorff_distance(gt_traj, pred_traj)
    rmse = _rmse_distance(gt_traj, pred_traj)

    if math.isinf(dfd) or math.isinf(hd) or math.isinf(rmse):
        print(f"[2D Eval] 返回 None 原因 -> 计算结果出现 inf")
        return None

    return {
        'Discrete_Fréchet_Distance': float(dfd),
        'Hausdorff_Distance': float(hd),
        'RMSE': float(rmse)
    }
    
# def evaluate_2d_trajectory(gt: Any, pred: Any) -> Optional[Dict[str, float]]:
#     """
#     评估轨迹相似度。如果失败，会在命令行打印原因并返回 None。
#     """
    
#     def _simple_extract_with_reason(obj: Any, name: str, traj_type: str = "2d") -> List[List[float]]:
#         key = f"{traj_type}_trajectory"
        
#         # 1. 检查对象是否为空
#         if obj is None:
#             print(f"[错误] {name} 数据为 None。")
#             return []
        
#         # 2. 检查字典格式
#         if isinstance(obj, dict):
#             if key not in obj:
#                 print(f"[错误] {name} 是字典但缺失键名: '{key}'。现有键: {list(obj.keys())}")
#                 return []
#             t = obj[key]
#         else:
#             # 假设直接是列表
#             t = obj

#         # 3. 检查是否为列表/元组
#         if not isinstance(t, (list, tuple)):
#             print(f"[错误] {name} 的轨迹数据不是列表格式，而是 {type(t)}。")
#             return []
            
#         # 4. 检查是否为空列表
#         if len(t) == 0:
#             print(f"[警告] {name} 的轨迹点列表为空。")
#             return []

#         # 5. 检查数据点格式 (假设 _is_number_list 已定义)
#         # 这里简单示范：检查第一个点是否为列表/元组
#         if not all(isinstance(pt, (list, tuple)) for pt in t):
#             print(f"[错误] {name} 轨迹中的数据点格式不正确（应为嵌套列表）。")
#             return []

#         return [list(map(float, pt)) for pt in t]

#     # 安全检查 gt 和 pred 是否有索引 0
#     try:
#         gt_item = gt[0]
#     except (IndexError, TypeError):
#         print("[错误] 无法获取 gt[0]，请检查 gt 是否为非空列表。")
#         return None

#     try:
#         pred_item = pred[0]
#     except (IndexError, TypeError):
#         print("[错误] 无法获取 pred[0]，请检查 pred 是否为非空列表。")
#         return None

#     # 提取轨迹并获取原因
#     gt_traj = _simple_extract_with_reason(gt_item, "Ground Truth")
#     pred_traj = _simple_extract_with_reason(pred_item, "Prediction")

#     if not gt_traj or not pred_traj:
#         # 原因已在子函数中打印
#         return None

#     # 计算指标
#     dfd = _discrete_frechet_distance(gt_traj, pred_traj)
#     hd = _hausdorff_distance(gt_traj, pred_traj)
#     rmse = _rmse_distance(gt_traj, pred_traj)

#     # 检查计算结果是否有效
#     metrics = {
#         'Discrete_Fréchet_Distance': dfd,
#         'Hausdorff_Distance': hd,
#         'RMSE': rmse
#     }
    
#     for k, v in metrics.items():
#         if math.isinf(v):
#             print(f"[错误] 指标 {k} 计算结果为无穷大(inf)，可能是轨迹点重合或数据异常。")
#             return None

#     return {k: float(v) for k, v in metrics.items()}


# ======================
# Vector Metrics
# ======================
def _extract_vectors_from(obj: Any) -> List[Tuple[Optional[str], List[float]]]:
    """Extract named vectors from obj similar to _extract_points_from.

    Supports formats like:
    - {'functional_vector': [-1, 1], 'label': 'pan_pour'}
    - [{'functional_vector': [[...]], 'label': 'pan_pour_passive'}]
    - plain list of numbers
    """
    out: List[Tuple[Optional[str], List[float]]] = []
    if obj is None:
        return out

    # single numeric vector
    if _is_number_list(obj):
        return [(None, list(obj))]

    if isinstance(obj, dict):
        # try common keys
        for k in ("functional_vector", "vector", "functional_vec"):
            if k in obj:
                v = obj[k]
                # v might be nested list [[...]] or single list
                if isinstance(v, (list, tuple)) and len(v) > 0 and _is_number_list(v[0]):
                    # nested list: take first
                    vec = v[0]
                    label = obj.get("label") or obj.get("name")
                    return [(label, list(vec))]
                if _is_number_list(v):
                    label = obj.get("label") or obj.get("name")
                    return [(label, list(v))]
        # fallback: scan values
        for vv in obj.values():
            if _is_number_list(vv):
                return [(None, list(vv))]

    if isinstance(obj, (list, tuple)):
        for item in obj:
            if item is None:
                continue
            if _is_number_list(item):
                out.append((None, list(item)))
                continue
            if isinstance(item, dict):
                for k in ("functional_vector", "vector", "functional_vec"):
                    if k in item:
                        v = item[k]
                        if isinstance(v, (list, tuple)) and len(v) > 0 and _is_number_list(v[0]):
                            out.append((item.get("label") or item.get("name"), list(v[0])))
                            break
                        if _is_number_list(v):
                            out.append((item.get("label") or item.get("name"), list(v)))
                            break
                else:
                    # fallback any numeric list value
                    for vv in item.values():
                        if _is_number_list(vv):
                            out.append((item.get("label") or item.get("name"), list(vv)))
                            break
        return out

    return out

def cosine_sim_single(gt: Any, pred: Any) -> Optional[float]:
    # For vectors, use vector extractor
    g = _extract_vectors_from(gt)
    p = _extract_vectors_from(pred)
    if not g or not p:
        return None

    # Single logic: take the first zip-paired element and handle nested structures
    for gv, pv in zip(g, p):
        gv_c = gv[1] if not isinstance(gv[1][0], list) else gv[1][0]
        pv_c = pv[1] if not isinstance(pv[1][0], list) else pv[1][0]
        return abs(_cosine(gv_c, pv_c))

    return None

def mean_cosine_sim(gt: Any, pred: Any) -> Optional[float]:
    g = _extract_vectors_from(gt)
    p = _extract_vectors_from(pred)
    if not g or not p:
        return None

    # Default: element-wise alignment using zip
    vals: List[float] = []
    for gv, pv in zip(g, p):
        # gv and pv are (label, vector). Handle nested structures similar to mean_l2_loss
        gv_c = gv[1] if not isinstance(gv[1][0], list) else gv[1][0]
        pv_c = pv[1] if not isinstance(pv[1][0], list) else pv[1][0]
        vals.append(abs(_cosine(gv_c, pv_c)))

    if not vals:
        return None
    return float(sum(vals) / len(vals))


# ======================
# Spatial Relation Metrics
# ======================
def spatial_rel_qa_obj2cam_distance(gt: Any, pred: Any) -> Optional[float]:
    """Metric for QA: absolute difference between GT and predicted distance to camera (meters)."""
    gt_d = _extract_distance_from_qa(gt)
    pred_d = _extract_distance_from_qa(pred)
    if gt_d is None or pred_d is None:
        return None
    return float(abs(gt_d - pred_d))

def spatial_rel_qa_obj2obj_distance(gt: Any, pred: Any) -> Optional[float]:
    """Metric for QA: absolute difference between GT and predicted object-to-object distance (meters)."""
    gt_d = _extract_distance_from_qa(gt)
    pred_d = _extract_distance_from_qa(pred)
    if gt_d is None or pred_d is None:
        return None
    return float(abs(gt_d - pred_d))


# ======================
# Choice Question Metrics
# ======================
def _extract_choice_from_qa(obj: Any) -> Optional[str]:
    """Extract a single-choice answer (like 'A'/'B' or fields named spatial_relation_choice) from QA objects.

    Handles:
    - simple string answers: 'A' or 'B' or ' "B" '
    - lists like [ { 'spatial_relation_choice': 'B', ... } ]
    - dicts with keys: 'spatial_relation_choice', 'choice', 'answer', 'value'
    - nested structures, JSON-in-string
    Returns uppercase single token (like 'A' or 'B' or other short textual choices) or None.
    """
    if obj is None:
        return None

    # direct string
    if isinstance(obj, str):
        s = obj.strip()
        # try parse JSON
        try:
            parsed = json.loads(s)
        except Exception:
            # common case: single letter or single token
            if len(s) == 1 and s.isalpha():
                return s.upper()
            # maybe the model returned '"B"' or "['B']" etc: try regex for single letter
            m = re.search(r"([A-Za-z])", s)
            if m:
                return m.group(1).upper()
            return None
        return _extract_choice_from_qa(parsed)

    # dict: check common keys
    if isinstance(obj, dict):
        # direct keys that may contain the choice
        for k in ("spatial_relation_choice", "spatial_relation", "choice", "answer", "selected", "selection", "value"):
            if k in obj:
                v = obj[k]
                if isinstance(v, str):
                    vv = v.strip()
                    if len(vv) == 1:
                        return vv.upper()
                    # try nested json in string
                    try:
                        parsed = json.loads(vv)
                    except Exception:
                        return vv
                    return _extract_choice_from_qa(parsed)
                if isinstance(v, (list, tuple)) or isinstance(v, dict):
                    res = _extract_choice_from_qa(v)
                    if res is not None:
                        return res
                # numeric or other: cast
                if isinstance(v, (int, float)):
                    return str(v)

        # sometimes responses nest under 'conversations' or 'value' etc
        for k in ("conversations", "value", "answers", "answer"):
            if k in obj:
                res = _extract_choice_from_qa(obj[k])
                if res is not None:
                    return res

        # fallback: search all values
        for vv in obj.values():
            res = _extract_choice_from_qa(vv)
            if res is not None:
                return res
        return None

    # list/tuple: look through items
    if isinstance(obj, (list, tuple)):
        for item in obj:
            if item is None:
                continue
            res = _extract_choice_from_qa(item)
            if res is not None:
                return res
        return None

    return None

def choice_accuracy(gt: Any, pred: Any) -> Optional[float]:
    """Return 1.0 if predicted choice equals GT choice, 0.0 if not, None if cannot extract."""
    print("-------------------")
    print("gt_choice:",gt)
    print("pred_choice:", pred)
    gt_choice = _extract_choice_from_qa(gt)
    pred_choice = _extract_choice_from_qa(pred)
    if gt_choice is None or pred_choice is None:
        return None
    try:
        return 1.0 if str(gt_choice).strip().upper() == str(pred_choice).strip().upper() else 0.0
    except Exception:
        return None


# Metric registry
METRIC_IMPL: Dict[str, Callable[[Any, Any], Optional[float]]] = {
    "l2": l2_loss_single,
    "mean_l2": mean_l2_loss,
    "cosine": cosine_sim_single,
    "mean_cosine": mean_cosine_sim,
    "oriented_bbox_2d": oriented_bbox_2d_metric,
    "obj_bbox_qa_detect": obj_bbox_qa_detect,
    "trajectory_3d": evaluate_3d_trajectory,
    "trajectory_2d": evaluate_2d_trajectory,
    "spatial_rel_qa_obj2cam_distance": spatial_rel_qa_obj2cam_distance,
    "spatial_rel_qa_obj2obj_distance": spatial_rel_qa_obj2obj_distance,
    "choice": choice_accuracy,
}


# Map task name patterns (exact or prefix) to metric keys
METRIC_BY_TASK: Dict[str, str] = {}

# ---------------------
# Pointing / functional point tasks
# ---------------------
METRIC_BY_TASK.update({
    "single_func_point_2d": "l2",
    "single_func_point_3d": "l2",
    "multi_func_point_2d": "mean_l2",
    "multi_func_point_3d": "mean_l2",
})

# Part-level pointing tasks (single -> l2, all/multi -> mean_l2)
METRIC_BY_TASK.update({
    "all_part_point_2d": "mean_l2",
    "all_part_point_3d": "mean_l2",
    "single_part_point_2d": "l2",
    "single_part_point_3d": "l2",
})

# Object-center QA (2D / 3D)
METRIC_BY_TASK.update({
    "object_point_qa_detect_all_2d": "mean_l2",
    "object_point_qa_detect_category_2d": "l2",
    "object_point_qa_single_with_desc_2d": "l2",
    "object_point_qa_detect_all_3d": "mean_l2",
    "object_point_qa_detect_category_3d": "l2",
    "object_point_qa_single_with_desc_3d": "l2",
})

# ---------------------
# Vector / functional vector tasks
# ---------------------
METRIC_BY_TASK.update({
    "single_func_vector_2d": "cosine",
    "single_func_vector_3d": "cosine",
    "multi_func_vector_2d": "mean_cosine",
    "multi_func_vector_3d": "mean_cosine",
})

# Grasp points
METRIC_BY_TASK.update({
    "grasp_point_2d": "l2",
    "grasp_point_3d": "l2",
})

# Correspondence pair tasks
METRIC_BY_TASK.update({
    "correspondence_pairs_single_object_2d": "l2",
    "correspondence_pairs_single_object_3d": "l2",
    "correspondence_pairs_customized_2d": "mean_l2",
    "correspondence_pairs_customized_3d": "mean_l2",
})

# Oriented 2D bounding boxes (all / category / single / parts)
METRIC_BY_TASK.update({
    "2d_oriented_bbox_qa_detect_all": "oriented_bbox_2d",
    "2d_oriented_bbox_qa_detect_category": "oriented_bbox_2d",
    "2d_oriented_bbox_qa_single_with_desc": "oriented_bbox_2d",
    "2d_oriented_bbox_qa_detect_all_parts": "oriented_bbox_2d",
    "2d_oriented_bbox_qa_single_part_with_desc": "oriented_bbox_2d",
})

# 3D bbox QA (full objects and parts)
METRIC_BY_TASK.update({
    "obj_bbox_qa_detect_all_3d": "obj_bbox_qa_detect",
    "obj_bbox_qa_detect_category_3d": "obj_bbox_qa_detect",
    "obj_bbox_qa_single_with_desc_3d": "obj_bbox_qa_detect",
    "3d_bbox_qa_detect_all_parts": "obj_bbox_qa_detect",
    "3d_bbox_qa_single_part_with_desc": "obj_bbox_qa_detect",
})

# Spatial relation QA
METRIC_BY_TASK.update({
    "spatial_rel_qa_obj2cam_distance": "spatial_rel_qa_obj2cam_distance",
    "spatial_rel_qa_obj2obj_distance": "spatial_rel_qa_obj2obj_distance",
    "spatial_rel_qa_obj2obj_choice": "choice",
})

# Trajectory QA
METRIC_BY_TASK.update({
    "traj_qa_prediction_2d": "trajectory_2d",
    "traj_qa_prediction_3d": "trajectory_3d",
})


def compute_metric(task_name: str, gt: Any, pred: Any) -> Dict[str, Optional[float]]:
    """Compute metric for given task_name between gt and pred.

    Returns dict: {"metric_name": str, "metric_value": float or None}
    """
    # choose metric key by exact match or prefix
    metric_key = None
    if task_name in METRIC_BY_TASK:
        metric_key = METRIC_BY_TASK[task_name]
    else:
        # try prefix matching
        for k in METRIC_BY_TASK:
            if task_name.startswith(k):
                metric_key = METRIC_BY_TASK[k]
                break

    if metric_key is None:
        return {"metric_name": None, "metric_value": None}

    func = METRIC_IMPL.get(metric_key)
    if func is None:
        return {"metric_name": metric_key, "metric_value": None}

    try:
        # print("gt:", gt)
        # print("pred:", pred)
        val = func(gt, pred)
        
    except Exception:
        val = None

    # If metric returned a scalar numeric value, coerce to float; otherwise return as-is (dicts, lists etc.)
    if val is None:
        metric_value = None
    elif isinstance(val, (int, float)):
        metric_value = float(val)
    else:
        metric_value = val

    return {"metric_name": metric_key, "metric_value": metric_value}
