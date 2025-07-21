# This file generates a random pose and converts it to a set of leg lengths.
import random
import math
import DQClass
import numpy as np
from scipy.spatial.transform import Rotation


############################
### Geometric Parameters ###
############################

scale = 1e-3  # Convert mm to meters

# Hexapod geometry (Some parameters are approximated or guessed for confidentiality)
d = 200 * scale       # Gap between adjacent table joints
D = 1759 * scale      # Table diameter
b = 150 * scale       # Gap between adjacent base joints
B = 2011.4 * scale    # Base diameter
h = 1171.7 * scale    # Table-top height


#############################
### Table and Base Points ###
#############################

# 60-degree rotation quaternion
R2 = DQClass.Quaternion(math.cos(math.pi / 3), 0, 0, math.sin(math.pi / 3))

# Table joints in identity pose
T1 = DQClass.ToVectorQuaternion([+d/2, D/2 - math.sqrt(3)*d/2, h])
T2 = DQClass.ToVectorQuaternion([-d/2, D/2 - math.sqrt(3)*d/2, h])
T3 = R2 * T1 * R2.conjugate()
T4 = R2 * T2 * R2.conjugate()
T5 = R2 * T3 * R2.conjugate()
T6 = R2 * T4 * R2.conjugate()
TABLE_POINTS = [T1, T2, T3, T4, T5, T6]

# Base joints in identity pose
B1 = DQClass.ToVectorQuaternion([+b/2, B/2 - math.sqrt(3)*b/2, 0])
B2 = DQClass.ToVectorQuaternion([-b/2, B/2 - math.sqrt(3)*b/2, 0])
B3 = R2 * B1 * R2.conjugate()
B4 = R2 * B2 * R2.conjugate()
B5 = R2 * B3 * R2.conjugate()
B6 = R2 * B4 * R2.conjugate()

# Here is the list of table coordinates as quaternions
TableID = [T1, T2, T3, T4, T5, T6]
# Here is the list of base coordinates as quaternions
BaseID = [B1, B2, B3, B4, B5, B6]

# Here are the limitations on the P3350 from the website.
MaxTranslation = 635 #mm (X,Y)
MaxHeave = 694 #mm (Z)
Pitch = 30 #degrees
Yaw = 45 #degrees



#######################
### Pose Generation ###
#######################

def MakeRandomPose(max_angle = .5, max_trans = .5):
    """
    Generate a random pose using dual quaternions.
    :param max_angle: Max rotation (radians)
    :param max_trans: Max translation (meters)
    :return: Dual quaternion pose
    """
    theta = random.uniform(-max_angle, max_angle)
    i = random.uniform(-1, 1)
    j = random.uniform(-1, 1)
    k = random.uniform(-1, 1)
    Q = (DQClass.Quaternion(0, i, j, k).normalization() * math.sin(1 / 2 * theta)
         + DQClass.Quaternion(math.cos(1 / 2 * theta), 0, 0, 0))
    t1 = random.uniform(-max_trans, max_trans)
    t2 = random.uniform(-max_trans, max_trans)
    t3 = random.uniform(0, max_trans)  # Table can't go below base
    t = DQClass.Quaternion(0, t1, t2, t3)
    RandomPose = DQClass.DQuaternion(Q, t * Q * (1/2))
    return RandomPose


def MakeEulerPose(max_angle = 0.5, max_trans = 0.5):
    """
    Generate a random 6-DoF pose as Euler angles + translation.
    :return: np.array([roll, pitch, yaw, x, y, z])
    """
    phi = random.uniform(-max_angle, max_angle)  # Roll
    theta = random.uniform(-max_angle, max_angle)  # Pitch
    psi = random.uniform(-max_angle, max_angle)  # Yaw
    t1 = random.uniform(-max_trans, max_trans)  # Translation X
    t2 = random.uniform(-max_trans, max_trans)  # Translation Y
    t3 = random.uniform(0, max_trans)  # Translation Z
    return np.array([phi, theta, psi, t1, t2, t3])


############################################
### Pose Evaluation (Inverse Kinematics) ###
############################################

def TableCoords(pose):
    """
    Compute the position of table points under a given pose (Q, t).
    """
    TablePosition = []
    Q = pose.A
    t = pose.B * pose.A.conjugate() * 2
    for n in range(6):
        r = DQClass.ToVectorQuaternion(TableID[n])
        s = Q * r * Q.conjugate() + t
        TablePosition.append([s.x, s.y, s.z])
    return np.array(TablePosition)


def BaseCoords(pose):
    """
    Compute the position of base points under a given pose.
    """
    BasePosition = []
    Q = pose.A
    t = pose.B * pose.A.conjugate() * 2
    for n in range(6):
        r = DQClass.ToVectorQuaternion(TableID[n])
        s = Q * r * Q.conjugate() + t
        BasePosition.append([s.x, s.y, s.z])
    return np.array(BasePosition)


def LegLengthsRedacted(pose):
    """
    Compute leg lengths for a given pose using dual quaternion transformation.
    """
    Q = pose.A
    t = pose.B * pose.A.conjugate() * 2
    Lengths = []
    for n in range(6):
        r = TableID[n]
        s = Q * r * Q.conjugate() + t
        basePoint = BaseID[n]
        L = (s-basePoint).norm()
        Lengths.append(L)
    return Lengths


def euler_lengths(pose_params, Table = TableID, Base = BaseID):
    """
    Compute leg lengths from Euler-angle-based pose.
    :param pose_params: [roll, pitch, yaw, x, y, z] in radians and meters
    :return: np.array of leg lengths
    """
    lengths = []
    phi, theta, psi, x, y, z = pose_params
    rot = Rotation.from_euler('zyx', [psi, theta, phi])
    rotation_matrix = rot.as_matrix()
    trans = np.array([x, y, z])
    for i in range(6):
        new_point = rotation_matrix.T @ (Base[i] - trans)
        lengths.append(np.linalg.norm(Table[i] - new_point))
    return np.array(lengths)
