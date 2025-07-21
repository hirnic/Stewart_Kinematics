"""This script solves the forward kinematics problem for Stewart platforms utilizing Euler angles. 
    Methods:
    - Standard Newton-Raphson method (Hirni)
    - Screw-theoretic Newton-Raphson method (Selig-Li)

    Modifiers:
    - Numerical method (Hirni)
    - Monotonic Descent (Zhu-Zhang)
    - Feed Forward Neural Network (Parikh-Lam)

    Honorable Mentions that were left out because they didn't work into the poster project very well:
    - Self-Tuning (Ren-Feng-Mills)
    - Levenburg-Marquardt method (Xie-Dai-Liu)

    We wish to count the number of operations (comparisons, bit shifts, additions, multiplications, divisions, square roots, exponents)
    and the number of iterations required to solve the problem with the given methods and modifiers.
"""

import numpy as np
import torch
from torch import nn
import RandomizerRedacted as rr
from scipy.spatial.transform import Rotation
from scipy.linalg import expm

def displacement_matrix(pose_params):
    """
    Computes the displacement matrix for a given pose in Euler angles.
    :param pose_params: np array of shape (6,) containing [phi, theta, psi, x, y, z]
    :return: 3x3 rotation matrix corresponding to the Euler angles.
    """
    phi, theta, psi = pose_params[:3]

    # Precompute trig terms
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    # Group terms to minimize repeated ops. Ideally these would be cached in such a way that retrieval is faster than recomputing.
    sth_sph = sth * sph
    sth_cph = sth * cph
    cth_sph = cth * sph
    cth_cph = cth * cph

    # Rotation matrix (3x3)
    R = np.array([
        [cps * cth,       cps * sth_sph - sps * cph,   cps * sth_cph + sps * sph],
        [sps * cth,       sps * sth_sph + cps * cph,   sps * sth_cph - cps * sph],
        [-sth,            cth_sph,                     cth_cph]
    ])

    return R


def square_inverse_kinematics_function(pose_params, table_points, base_points):
    """
    Computes the squared leg lengths for a Stewart platform given the pose parameters and points on the table and base.
    :param pose_params: np array of shape (6,) containing [phi, theta, psi, x, y, z]
    :param table_points: np array of shape (6, 3) containing points on the table in the identity pose.
    :param base_points: np array of shape (6, 3) containing points on the base in the identity pose.
    :return: np array of shape (6,) containing the squared leg lengths.
    """
    R = displacement_matrix(pose_params)

    # Avoid Python for-loop by vectorizing the transform
    transformed_points = (R @ table_points.T).T  + pose_params[3:]# Shape: (6, 3)

    diffs = base_points - transformed_points      # Shape: (6, 3)
    length_squares = np.einsum('ij,ij->i', diffs, diffs)

    return length_squares


def symbolic_jacobian(pose_params, table_points, base_points):
    """
    Computes the Jacobian of the squared leg lengths with respect to the pose parameters.
    :param pose_params: np array of shape (6,) containing [phi, theta, psi, x, y, z]
    :param table_points: np array of shape (6, 3) containing points on the table in the identity pose.
    :param base_points: np array of shape (6, 3) containing points on the base in the identity pose.
    :return: np array of shape (6, 6) representing the Jacobian matrix.
    """
    phi, theta, psi, x, y, z = pose_params

    # Precompute trig functions
    cph, sph = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cps, sps = np.cos(psi), np.sin(psi)

    # Precompute trig products
    sth_sph = sth * sph
    sth_cph = sth * cph
    cth_sph = cth * sph
    cth_cph = cth * cph
    sps_sph = sps * sph
    sps_cph = sps * cph
    cps_sph = cps * sph
    cps_cph = cps * cph
    cps_sth = cps * sth
    sps_sth = sps * sth
    cps_cth = cps * cth
    sps_cth = sps * cth
    cth_cph = cth * cph
    cth_cph = cth * cph

    # Rotation matrix R
    R = np.array([
        [cps * cth,       cps_sth * sph - sps_cph,   cps_sth * cph + sps_sph],
        [sps * cth,       sps_sth * sph + cps_cph,   sps_sth * cph - cps_sph],
        [-sth,            cth_sph,                   cth_cph]
    ])

    # Derivatives
    dR_dphi = np.array([
        [0,                cps_sth * cph + sps_sph,   sps_cph - cps_sth * sph],
        [0,                sps_sth * cph - cps_sph,  -sps_sth * sph - cps_cph],
        [0,                cth_cph,                  -cth_sph]
    ])

    dR_dtheta = np.array([
        [-cps_sth,   cps_cth * sph,  cps_cth * cph],
        [-sps_sth,   sps_cth * sph,  sps_cth * cph],
        [-cth,      -sth_sph,       -sth_cph]
    ])

    dR_dpsi = np.array([
        [-sps_cth,  -sps_sth * sph - cps_cph,   cps_sph - sps_sth * cph],
        [cps_cth,    cps_sth * sph - sps_cph,   cps_sth * cph + sps_sph],
        [0,          0,                         0]
    ])

    t = np.array([x, y, z])
    Jacobian = []
    for i in range(6):
        tp = table_points[i]
        bp = base_points[i]

        Rp = R @ tp
        diff = Rp + t - bp

        dRp_dphi   = dR_dphi   @ tp
        dRp_dtheta = dR_dtheta @ tp
        dRp_dpsi   = dR_dpsi   @ tp

        J_row = [diff @ dRp_dphi,diff @ dRp_dtheta,diff @ dRp_dpsi,diff[0],diff[1],diff[2]]
        Jacobian.append(J_row)

    return 2*np.array(Jacobian)



def numerical_jacobian(pose_params, table_points, base_points, epsilon=1e-6):
    """
    Computes the numerical Jacobian of the squared leg lengths with respect to the pose parameters.
    :param pose_params: np array of shape (6,) containing [phi, theta, psi, x, y, z]
    :param table_points: np array of shape (6, 3) containing points on the table in the identity pose.
    :param base_points: np array of shape (6, 3) containing points on the base in the identity pose.
    :param epsilon: Small value for numerical differentiation.
    :return: np array of shape (6, 6) representing the Jacobian matrix.
    """
    J = []
    for i in range(6):
        delta = np.zeros(6)
        delta[i] = epsilon
        pos_plus = square_inverse_kinematics_function(pose_params + delta, table_points, base_points)
        pos_minus = square_inverse_kinematics_function(pose_params - delta, table_points, base_points)
        J.append((pos_plus - pos_minus) / (2 * epsilon))
    return np.array(J).T


def standard_pose_finder(init, lengths, TableID, Base, max_iterations=25, monotonic_descent=False, numerical=False):
    
    """
    Given the table data and leg lengths, find the pose of the Stewart platform using the standard Newton-Raphson method.

    :param init: an engineer's pose (array of 6 floats: phi,theta,psi, rotation; x,y,z, translation). Initial guess.
    :param lengths: array of 6 floats. Leg lengths.
    :param TableID: array of arrays containing the coordinates of the 6 points on the table where the legs attach.
    :param Base: array of arrays containing the coordinates of the 6 points on the base where the legs attach.
    :return: An array of 6 floats representing the Euler angles and translations that solve the FK problem.
    """
    try:
        iters = 0
        converged = True
        U = init
        length_squares = [l * l for l in lengths]
        while True:
            DL = square_inverse_kinematics_function(U, TableID, Base)
            DL = [dl - ls for dl, ls in zip(DL, length_squares)]
            if np.max(np.abs(DL)) < 1e-6:
                break

            # I discovered that this next step has room for optimization. 
            # We can cut out a few trig ops and a few dozen x/+ ops by computing the Jacobian with the square_inverse_kinematics.
            # Oops.
            J = numerical_jacobian(U, TableID, Base) if numerical else symbolic_jacobian(U, TableID, Base)  
            DU = np.linalg.solve(J, DL)

            if monotonic_descent:
                descent_param = 1.0
                while True:
                    X = U - DU * descent_param
                    loss_trial = square_inverse_kinematics_function(X, TableID, Base)
                    DL_trial = [lt - ls for lt, ls in zip(loss_trial, length_squares)]
                    if np.max(np.abs(DL_trial)) <= np.max(np.abs(DL)) or descent_param < 1e-4:
                        break
                    descent_param *= 0.5
                U = X
                DL = DL_trial
            else:
                U = U - DU

            iters += 1
            if iters == 10:
                converged = False
                break

    except Exception:
        converged = False

    return U, iters, converged


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################


def rigid_body_matrix(pose_params):
    """
    Converts pose parameters into a 4x4 rigid body transformation matrix.
    :param pose_params: np array of shape (6,) containing [phi, theta, psi, x, y, z]
    :return: 4x4 np.array representing the rigid body transformation matrix.
    """
    x, y, z = pose_params[3:]
    R = displacement_matrix(pose_params)
    M_scipy = np.eye(4)
    M_scipy[:3, :3] = R
    M_scipy[:3, 3] = [x, y, z]
    return M_scipy


def screw_matrix(screw_vec):
    """
    :param screw_vec: an np.array with 6 elements [w1, w2, w3, v1, v2, v3]
    :return: 4x4 np.array representing the se(3) Lie algebra element
    """
    w1, w2, w3, v1, v2, v3 = screw_vec
    return np.array([
        [0, -w3,  w2, v1],
        [w3,  0, -w1, v2],
        [-w2, w1,  0, v3],
        [0,   0,   0,  0]
    ])


def screw_inverse_kinematics_function(pose_matrix, table_points, base_points):
    """
    :param pose_params: np array [phi, theta, psi, x, y, z]
    The Euler angles are phi, theta, psi, (radians) and translations are x, y, z (millimeters).
    :param table: The point on the table in the identity pose (np array)
    :param base: the point on the base in the identity pose (np array).
    :return: np array of floats (the six squared leg lengths in millimeters). Leg lengths.
    """
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]

    displaced = (R @ table_points.T).T + t
    diffs = base_points - displaced
    return np.sum(diffs**2, axis=1)


def screw_jacobian(pose_matrix, table_points, base_points):
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]

    b_prime = (R @ table_points.T).T + t
    crosses = np.cross(base_points, b_prime)
    diffs = b_prime - base_points
    J = np.hstack((crosses, diffs))
    return 2 * J


def screw_pose_finder(init, lengths, TableID, Base, max_iters = 25, monotonic_descent = False):
    """
    :param init: an engineer's pose (array of 6 floats: phi,theta,psi, rotation; x,y,z, translation). Initial guess.
    :param lengths: array of 6 floats. Leg lengths.
    :param TableID: array of arrays containing the coordinates of the 6 points on the table where the legs attach.
    :param Base: array of arrays containing the coordinates of the 6 points on the base where the legs attach.
    :return: An element of the Lie Group SE(3) (4 x 4 np.array of the form ((R, v), (0, 1)))
    """
    iters = 0
    converged = True
    M = rigid_body_matrix(init)
    square_lengths = np.array(lengths)**2
    
    while True:
        L0 = screw_inverse_kinematics_function(M, TableID, Base)
        DL = L0 - square_lengths
        if np.max(np.abs(DL)) < 1e-6:
            break
        current_loss = np.max(np.abs(DL))
        J = screw_jacobian(M, TableID, Base)
        screw_vec = np.linalg.solve(J, DL) * (-1)
        X_new = expm(screw_matrix(screw_vec)) @ M  # CAN USE RODRIGUES'S EXPONENTIAL MAP. Operations were counted as though we were using Rodrigues.

        if monotonic_descent:
            descent_param = 1.0
            while True:
                loss = screw_inverse_kinematics_function(X_new, TableID, Base) - square_lengths
                if np.max(np.abs(loss)) < current_loss or descent_param < 1e-4:
                    break
                descent_param *= 0.5
                X_new = expm(screw_matrix(screw_vec * descent_param)) @ M  # CAN USE RODRIGUES'S EXPONENTIAL MAP
            M = X_new
        else:
            M = X_new

        iters += 1
        if iters >= max_iters:
            converged = False
            break
    return M, iters, converged


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################


# define a simple feedforward neural network used for initial estimates.
class stewartnet_1_layer(nn.Module):
    def __init__(self, a, output_layer = 6):
        self.a = a
        self.out = output_layer
        super().__init__()
        self.model = nn.Sequential(
        nn.Linear(6, self.a),
        nn.ReLU(),
        nn.Linear(self.a, self.out)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, a=-0.2, b=0.2)
                nn.init.uniform_(m.bias, a=-0.2, b=0.2)

    def forward(self, x):
        return self.model(x)


class Euler_Class_Pose_Finder:
    def __init__(self, standard_NR = False, screw_NR = False, screw_LM = False, numerical = False,
                monotonic_descent = False, FFNN = False, domain = 1):
        self.standard_NR = standard_NR
        self.screw_NR = screw_NR
        self.screw_LM = screw_LM
        self.numerical = numerical
        self.monotonic_descent = monotonic_descent
        self.FFNN = FFNN
        self.table = np.array([rr.TableID[i].ToPureVec() for i in range(6)])
        self.base = np.array([rr.BaseID[i].ToPureVec() for i in range(6)])
        self.num_samples = 100
        self.domain = domain
        self.numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
        self.poses, self.leg_lengths = self.generate_sample_set()

    def generate_sample_set(self, domain = None):
        """
        Generate a set of random poses for the Stewart platform.

        param num_samples: Number of random poses to generate
        param max_angle: Maximum angle in radians for the Euler angles (phi, theta, psi)
        param max_trans: Maximum translation in meters for the translations (x, y, z)
        return: np arrays of shape (num_samples, 6) for poses and leg lengths.
        """
        if domain is None:
            domain = self.domain / 10
        print("Generating Samples...")
        poses = []
        lengths = []
        for _ in range(self.num_samples):
            pose = rr.MakeEulerPose(domain, domain)
            poses.append(pose)
            leg_length = rr.euler_lengths(pose, self.table, self.base)
            lengths.append(leg_length)
        return np.array(poses), torch.tensor(np.array(lengths), dtype=torch.float32)

    def solve(self):
        """
        Solve the forward kinematics problem for a Stewart platform using Euler angles.
        
        :param leg_lengths: np array of leg lengths
        :param table_points: np array of points on the table in the identity pose
        :param base_points: np array of points on the base in the identity pose
        :param initial_guess: Initial guess for the Euler angles and translations [phi, theta, psi, x, y, z]
        :return: np array of Euler angles and translations that solve the FK problem
        """
        # Check if exactly one of self.standard_NR, self.screw_NR, self.gauss_newton is True
        if self.standard_NR + self.screw_NR + self.screw_LM != 1:
            raise ValueError("Exactly one of standard_NR, screw_NR, or gauss_newton must be True.")
            return
        failed = 0
        num_samples = self.num_samples
        model = None
        initial_guess = np.zeros(6)

        if self.FFNN:
            model = stewartnet_1_layer(20)
            model.load_state_dict(torch.load('Neural_Net_Parameters\Euler_Angles\s_20_' + str(self.numerals[self.domain - 1]) + '.pth'))
            model.eval()

        if self.standard_NR:
            iterations = 0
            for length in self.leg_lengths:
                if self.FFNN:
                    initial_guess = model(length.float().unsqueeze(0)).detach().cpu().numpy().flatten()
                pose, iters, converged = standard_pose_finder(initial_guess, length, self.table, self.base,
                                                      monotonic_descent = self.monotonic_descent, numerical = self.numerical)
                if not converged:
                    num_samples -= 1
                    failed += 1
                else:
                    iterations += iters
            return iterations/num_samples, failed

        if self.screw_NR:
            iterations = 0
            for length in self.leg_lengths:
                if self.FFNN:
                    initial_guess = model(length.float().unsqueeze(0)).detach().cpu().numpy().flatten()
                pose, iters, converged = screw_pose_finder(initial_guess, length, self.table, self.base,
                                                      monotonic_descent = self.monotonic_descent)
                if not converged:
                    num_samples -= 1
                    failed += 1
                else:
                    iterations += iters
                
            return (iterations/num_samples, failed) if num_samples > 0 else (0, failed)



from wakepy import keep

with keep.running():
    for i in range(1, 11):
        print(f"Domain: {i}")
        pose_finder = Euler_Class_Pose_Finder(standard_NR = True, domain = i)
        print("Standard Newton-Raphson Method")
        for i in range(2):
            pose_finder.monotonic_descent = i
            for k in range(2):
                pose_finder.FFNN = k
                for j in range(2):
                    pose_finder.numerical = j
                    print(f"Monotonic Descent: {bool(pose_finder.monotonic_descent)} - FFNN: {bool(pose_finder.FFNN)} - Numerical: {bool(pose_finder.numerical)}")
                    print(pose_finder.solve())
        pose_finder.standard_NR = False
        pose_finder.screw_NR = True
        print("\n Lie-Based Newton-Raphson Method")
        for i in range(2):
            pose_finder.monotonic_descent = i
            for k in range(2):
                pose_finder.FFNN = k
                print(f"Monotonic Descent: {bool(pose_finder.monotonic_descent)} - FFNN: {bool(pose_finder.FFNN)}")
                print(pose_finder.solve())
        print("\n")
        print("-------------------------------------------------------------------")
        print("\n")
