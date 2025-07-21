"""This script solves the forward kinematics problem for Stewart platforms utilizing dual quaternions. 
    Methods:
    - Standard Newton-Raphson method (Hirni)
    - Screw-theoretic Newton-Raphson method (Montgomery_Smith-Shy)

    Modifiers:
    - Monotonic Descent (Zhu-Zhang)
    - Self-Tuning (Ren-Feng-Mills)
    - Feed Forward Neural Network (Parikh-Lam)

    We wish to count the number of operations (comparisons, bit shifts, additions, multiplications, divisions, square roots, exponents)
    and the number of iterations required to solve the problem with the given methods and modifiers.
"""

import numpy as np
import random
import math
import torch
from torch import nn
import DQClass
import RandomizerRedacted as rr


def numerical_jacobian(X, lengths, TableID, Base, epsilon=1e-5):
    """
    Computes the numerical Jacobian of the loss function with respect to the parameters of the dual quaternion.

    :param X: The dual quaternion representing the pose of the table.
    :param lengths: The lengths of the legs of the Stewart platform.
    :param TableID: The coordinates of the table in the identity pose.
    :param Base: The coordinates of the base in the identity pose.
    :param epsilon: The perturbation value for finite difference approximation.
    :return: The Jacobian matrix of the loss function with respect to the parameters.
    """
    params = np.array([X.A.x, X.A.y, X.A.z, X.B.x, X.B.y, X.B.z], dtype=float)
    J_columns = []
    lengths_squared = lengths ** 2
    def compute_loss(params):
        Q1, Q2, Q3, t1, t2, t3 = params
        Q0_sq = 1.0 - Q1**2 - Q2**2 - Q3**2
        Q0 = np.sqrt(Q0_sq)
        A = DQClass.Quaternion(Q0, Q1, Q2, Q3)
        B = DQClass.Quaternion(0, t1, t2, t3)
        A_conj = A.conjugate()
        loss = []
        for n in range(6):
            r = TableID[n]
            s = A * r * A_conj + B
            b = Base[n]
            u = s - b
            loss.append(u.norm_sq() - lengths_squared[n])
        return np.array(loss)

    for j in range(6):
        params_perturb_back1 = params.copy()
        params_perturb_fwd1 = params.copy()
        params_perturb_back1[j] -= epsilon
        params_perturb_fwd1[j] += epsilon
        f_back1 = compute_loss(params_perturb_back1)
        f_fwd1 = compute_loss(params_perturb_fwd1)
        column = (f_fwd1 - f_back1) / (2 * epsilon)
        J_columns.append(column)
    J = np.stack(J_columns, axis=1)
    return J


def compute_quaternion_loss(X, lengths, TableID, Base):
    """
    Computes the loss function for the dual quaternion pose of the table.
    :param X: The dual quaternion representing the pose of the table.
    :param lengths: The lengths of the legs of the Stewart platform.
    :param TableID: The coordinates of the table in the identity pose.
    :param Base: The coordinates of the base in the identity pose.
    :return: The loss vector, which is the difference between the squared leg lengths and the squared predicted lengths
    """
    length_sq = lengths ** 2
    Q0 = X.A.w
    Q1 = X.A.x
    Q2 = X.A.y
    Q3 = X.A.z
    t1 = X.B.x
    t2 = X.B.y
    t3 = X.B.z

    Q0Q1, Q0Q2, Q0Q3= Q0 * Q1, Q0 * Q2, Q0 * Q3
    Q1Q2, Q1Q3 = Q1 * Q2, Q2 * Q3
    Q2Q3 = Q2 * Q3

    Q02 = Q0 ** 2
    Q12 = Q1 ** 2
    Q22 = Q2 ** 2
    Q32 = Q3 ** 2
    lossVec = []
    for n in range(6):
        r = TableID[n]
        A = 2 * (r.y * (Q1Q2 - Q0Q3) + r.z * (Q1Q3 + Q0Q2)) + r.x * (1 - 2 * Q22 - 2 * Q32) + t1
        B = 2 * (r.x * (Q1Q2 + Q0Q3) + r.z * (Q2Q3 - Q0Q1)) + r.y * (1 - 2 * Q12 - 2 * Q32) + t2
        C = 2 * (r.x * (Q1Q3 - Q0Q2) + r.y * (Q2Q3 + Q0Q1)) + r.z * (1 - 2 * Q12 - 2 * Q22) + t3
        s = DQClass.Quaternion(0, A, B, C)
        b = Base[n]
        u = s - b
        lossVec.append(u.norm_sq() - length_sq[n])
    return np.array(lossVec)


def standard_pose_finder(init, lengths, TableID, Base, max_iterations=25, monotonic_descent = False, numerical = False):
    """
    Solves the forward kinematics problem for a Stewart platform using the standard Newton-Raphson method.
    :param init: Initial guess for the dual quaternion representing the pose of the table.
    :param lengths: The lengths of the legs of the Stewart platform.
    :param TableID: The coordinates of the table in the identity pose.
    :param Base: The coordinates of the base in the identity pose.
    :return: A tuple containing the dual quaternion representing the pose of the table, the number of iterations taken, and a boolean indicating convergence.
    """
    converged = True
    X, Y = init, DQClass.ZeroDQ()
    iters=0
    length_sq = lengths ** 2
    try:
        while True:
            Y = X
            Q0 = X.A.w
            Q1 = X.A.x
            Q2 = X.A.y
            Q3 = X.A.z
            t1 = X.B.x
            t2 = X.B.y
            t3 = X.B.z

            Q0Q1, Q0Q2, Q0Q3= Q0 * Q1, Q0 * Q2, Q0 * Q3
            Q1Q2, Q1Q3 = Q1 * Q2, Q2 * Q3
            Q2Q3 = Q2 * Q3

            Q02 = Q0 ** 2
            Q12 = Q1 ** 2
            Q22 = Q2 ** 2
            Q32 = Q3 ** 2

            # Here we extract the Jacobian Matrix
            Jacobian = []
            lossVec = []
            for n in range(6):
                r = TableID[n]
                A = 2 * (r.y * (Q1Q2 - Q0Q3) + r.z * (Q1Q3 + Q0Q2)) + r.x * (1 - 2 * Q22 - 2 * Q32) + t1
                B = 2 * (r.x * (Q1Q2 + Q0Q3) + r.z * (Q2Q3 - Q0Q1)) + r.y * (1 - 2 * Q12 - 2 * Q32) + t2
                C = 2 * (r.x * (Q1Q3 - Q0Q2) + r.y * (Q2Q3 + Q0Q1)) + r.z * (1 - 2 * Q12 - 2 * Q22) + t3
                s = DQClass.Quaternion(0, A, B, C)
                b = Base[n]
                u = s - b
                lossVec.append(u.norm_sq() - length_sq[n])

                # Here are partial derivatives involved in the length function derived from formulas
                DADQ1 = 2 * (r.y * (Q0Q2 + Q1Q3) + r.z * (Q0Q3 - Q1Q2))
                DADQ2 = 2 * (r.y * (Q0Q1 + Q2Q3) + r.z * (Q02 - Q22) - 2 * r.x * Q0Q2)
                DADQ3 = 2 * (r.y * (Q32 - Q02) + r.z * (Q0Q1 - Q2Q3) - 2 * r.x * Q0Q3)
                DBDQ1 = 2 * (r.x * (Q0Q2 - Q1Q3) + r.z * (Q12 - Q02) - 2 * r.y * Q0Q1)
                DBDQ2 = 2 * (r.x * (Q0Q1 - Q2Q3) + r.z * (Q0Q3 + Q1Q2))
                DBDQ3 = 2 * (r.x * (Q02 - Q32) + r.z * (Q0Q2 + Q1Q3) - 2 * r.y * Q0Q3)
                DCDQ1 = 2 * (r.x * (Q0Q3 + Q1Q2) + r.y * (Q02 - Q12) - 2 * r.z * Q0Q1)
                DCDQ2 = 2 * (r.x * (Q22 - Q02) + r.y * (Q0Q3 - Q1Q2) - 2 * r.z * Q0Q2)
                DCDQ3 = 2 * (r.x * (Q0Q1 + Q2Q3) + r.y * (Q0Q2 - Q1Q3))
                DSDQ1 = DQClass.Quaternion(0, DADQ1, DBDQ1, DCDQ1)
                DSDQ2 = DQClass.Quaternion(0, DADQ2, DBDQ2, DCDQ2)
                DSDQ3 = DQClass.Quaternion(0, DADQ3, DBDQ3, DCDQ3)


                DFDQ1 = (2 * u.DotProduct(DSDQ1)) / Q0
                DFDQ2 = (2 * u.DotProduct(DSDQ2)) / Q0
                DFDQ3 = (2 * u.DotProduct(DSDQ3)) / Q0
                DFDT1 = 2 * u.x
                DFDT2 = 2 * u.y
                DFDT3 = 2 * u.z

                Jacobian.append(np.array([DFDQ1, DFDQ2, DFDQ3, DFDT1, DFDT2, DFDT3]))

            lossVec = np.array(lossVec)
            if np.max(np.abs(lossVec)) < 1e-6:
                break
            matrix = np.array(Jacobian)
            if numerical:
                matrix = numerical_jacobian(X, lengths, TableID, Base)
            addend = np.linalg.solve(matrix, lossVec)

            X_new = X.To6Vec() - addend
            rotation = DQClass.ToQuaternionRotation(X_new[:3])
            translation = DQClass.Quaternion(0, X_new[3], X_new[4], X_new[5])
            X_new = DQClass.DQuaternion(rotation, translation)

            if monotonic_descent:
                descent_param = 1
                X_candidate = X_new
                while True:
                    loss = compute_quaternion_loss(X_candidate, lengths, TableID, Base)
                    if (np.max(np.abs(loss)) < np.max(np.abs(lossVec))) or descent_param < 1e-4:
                        break
                    descent_param *= 0.5
                    if descent_param == 1:
                        X_old = DQClass.DQuaternion(X.A, X.B * X.A * 0.5)  # This is the previous pose dual quaternion
                        X_new = DQClass.DQuaternion(X_new.A, X_new.B * X_new.A * 0.5)  # This is the new pose dual quaternion
                    X_candidate = X_old.slerp(X_new, descent_param)
                    X_candidate = DQClass.DQuaternion(X_candidate.A, X_candidate.B * X_candidate.A.conjugate()* 2) # convert back to my basic quaternion representation
                X = X_candidate
            else:
                X = X_new
            
            iters+=1
            if iters == max_iterations:
                converged = False
                break

        X.B = X.B * X.A * 0.5

    except Exception as e:
        converged = False

    return X, iters, converged


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################


def compute_screw_loss(X, lengths, TableID, Base):
    """
    Computes the loss function for the dual quaternion pose of the table using the screw-theoretic method.
    :param X: The dual quaternion representing the pose of the table.
    :param lengths: The lengths of the legs of the Stewart platform.
    :param TableID: The coordinates of the table in the identity pose.
    :param Base: The coordinates of the base in the identity pose.
    :return: The loss vector, which is the difference between the squared leg lengths and the squared predicted lengths.
    """
    f= []
    A_conj = X.A.conjugate()
    B_conj = X.B.conjugate()
    for n in range(6):
        s = (A_conj * Base[n] + B_conj * 2) * X.A
        f.append((TableID[n] - s).norm() - lengths[n])
    return f

def screw_pose_finder(init, lengths, TableID, Base, max_iters = 25, monotonic_descent = False):
    """
    Solves the forward kinematics problem for a Stewart platform using the screw-theoretic Newton-Raphson method.
    :param init: Initial guess for the dual quaternion representing the pose of the table.
    :param lengths: The lengths of the legs of the Stewart platform.
    :param TableID: The coordinates of the table in the identity pose.
    :param Base: The coordinates of the base in the identity pose.
    :return: A tuple containing the dual quaternion representing the pose of the table, the number of iterations taken, and a boolean indicating convergence.
    """
    try:
        converged = True
        X, Y = init, DQClass.ZeroDQ()
        iters = 0
        while True:
            Y = X
            f, Lf = [], []
            A_conj = X.A.conjugate()
            B_conj = X.B.conjugate()
            for n in range(6):
                s = (A_conj * Base[n] + B_conj * 2) * X.A                   # Translated points in moving frame
                f.append((TableID[n] - s).norm() - lengths[n])              # Loss function
                u = np.array((TableID[n] - s).normalization().ToPureVec())  # Unit vectors in direction of legs
                Cross = np.cross(np.array(TableID[n].ToPureVec()), u)       # Part of the Lie derivative
                Lf.append(np.concatenate([Cross, u]) * 2)                   # Lie Derivative
            if np.max(np.abs(np.array(f))) < 1e-6:
                break
            Theta = np.linalg.solve(np.array(Lf), np.array(f)) * (-1)
            Theta = DQClass.ToVectorDualQuaternion(Theta)                   #Screw
            Hat = (DQClass.IdentityDQ() + Theta).normalization()            #Less expensive version of the exponential map
            X_new = X * Hat

            if monotonic_descent:
                descent_param = 1
                X_candidate = X_new
                while True:
                    loss = compute_screw_loss(X_candidate, lengths, TableID, Base)
                    if np.max(np.abs(loss)) < np.max(np.abs(f)) or descent_param < 1e-4:
                        break
                    descent_param *= 0.5
                    X_candidate = X.slerp(X_new, descent_param)
                X = X_candidate
            else:
                X = X_new
            iters += 1

            if iters == max_iters:
                converged = False
                break
    except Exception as e:
        converged = False
    return X, iters, converged


#########################################################################################################################################
#########################################################################################################################################
#########################################################################################################################################


# define a simple feed-forward neural network to estimate the initial guess for the pose of the table
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


class Quaternion_Class_Pose_Finder:
    def __init__(self, standard_NR = False, screw_NR = False, numerical = False,
                monotonic_descent = False, FFNN = False, domain = 1):
        self.standard_NR = standard_NR
        self.screw_NR = screw_NR
        self.numerical = numerical
        self.monotonic_descent = monotonic_descent
        self.FFNN = FFNN
        self.table = rr.TableID
        self.base = rr.BaseID
        self.domain = domain
        self.numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
        self.num_samples = 100
        self.poses, self.leg_lengths = self.generate_sample_set()

    def generate_sample_set(self, domain = None):
        """
        Generate a set of random poses for the Stewart platform.

        param num_samples: Number of random poses to generate
        param max_angle: Maximum angle in radians for the Euler angles (phi, theta, psi)
        param max_trans: Maximum translation in meters for the translations (x, y, z)
        return: np arrays of shape (num_samples, 6) for poses and leg lengths.
        """
        print("Generating Samples...")
        if domain is None:
            domain = self.domain
        poses = []
        lengths = []
        for _ in range(self.num_samples):
            RandomPose = rr.MakeRandomPose(0.1*domain)
            leg_length = rr.LegLengthsRedacted(RandomPose)
            poses.append(RandomPose)
            lengths.append(leg_length)
        return poses, torch.tensor(np.array(lengths), dtype=torch.float32)

    def solve(self):
        """
        Solve the forward kinematics problem for a Stewart platform using Euler angles.
        
        :param leg_lengths: np array of leg lengths
        :param table_points: np array of points on the table in the identity pose
        :param base_points: np array of points on the base in the identity pose
        :param initial_guess: Initial guess for the Euler angles and translations [phi, theta, psi, x, y, z]
        :return: np array of Euler angles and translations that solve the FK problem
        """
        num_samples = self.num_samples
        if self.standard_NR + self.screw_NR != 1:
            raise ValueError("Exactly one of standard_NR or screw_NR must be True.")
            return
        failed = 0
        model = None
        initial_guess = DQClass.IdentityDQ()

        if self.FFNN:
            model = stewartnet_1_layer(17, 8)
            model.load_state_dict(torch.load('Neural_Net_Parameters\Dual_Quaternions\s_17_' + str(self.numerals[self.domain-1]) +'.pth'))
            model.eval()

        if self.standard_NR:
            iterations = 0
            for length in self.leg_lengths:
                if self.FFNN:
                    initial_guess = DQClass.ToDualQuaternion(model(length.float().unsqueeze(0)).detach().cpu().numpy().flatten()).normalization()
                pose, iters, converged = standard_pose_finder(initial_guess, length, self.table, self.base,
                                                      monotonic_descent = self.monotonic_descent, numerical = self.numerical)
                if not converged:
                    num_samples -= 1
                    failed += 1
                else:
                    iterations += iters
            return (iterations/num_samples, failed) if num_samples > 0 else (0, failed)

        if self.screw_NR:
            iterations = 0
            for length in self.leg_lengths:
                if self.FFNN:
                    initial_guess = DQClass.ToDualQuaternion(model(length.float().unsqueeze(0)).detach().cpu().numpy().flatten()).normalization()
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
    for domain in range(8, 11):
        print(f"Domain: {domain}")
        pose_finder = Quaternion_Class_Pose_Finder(standard_NR = True, domain = domain)
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

# pose_finder = Quaternion_Class_Pose_Finder(standard_NR = True, monotonic_descent=True, domain = 8)
# for k in range(2):
#     pose_finder.FFNN = k
#     for j in range(2):
#         pose_finder.numerical = j
#         print(f"Monotonic Descent: {bool(pose_finder.monotonic_descent)} - FFNN: {bool(pose_finder.FFNN)} - Numerical: {bool(pose_finder.numerical)}")
#         print(pose_finder.solve())



        # if monotonic_descent:
        #         descent_param = 1
        #         X = X.To6Vec() - addend
        #         Q1 = X[0]
        #         Q2 = X[1]
        #         Q3 = X[2]
        #         Q02 = 1 - Q1**2 - Q2**2 - Q3**2
        #         Q0 = np.sqrt(Q02)  # cos(alpha/2)
        #         cosao4 = np.sqrt((1+Q0)/2)  # cos(alpha/4)
        #         sinao4 = np.sqrt((1-Q0)/2)  # sin(alpha/4)
        #         Q0Q1, Q0Q2, Q0Q3= Q0 * Q1, Q0 * Q2, Q0 * Q3
        #         Q1Q2, Q1Q3 = Q1 * Q2, Q2 * Q3
        #         Q2Q3 = Q2 * Q3
        #         Q12 = Q1 ** 2
        #         Q22 = Q2 ** 2
        #         Q32 = Q3 ** 2
        #         t1 = X[3]
        #         t2 = X[4]
        #         t3 = X[5]
        #         loss = []
        #         for n in range(6):
        #             r = TableID[n]
        #             A = 2 * (r.y * (Q1Q2 - Q0Q3) + r.z * (Q1Q3 + Q0Q2)) + r.x * (1 - 2 * Q22 - 2 * Q32) + t1
        #             B = 2 * (r.x * (Q1Q2 + Q0Q3) + r.z * (Q2Q3 - Q0Q1)) + r.y * (1 - 2 * Q12 - 2 * Q32) + t2
        #             C = 2 * (r.x * (Q1Q3 - Q0Q2) + r.y * (Q2Q3 + Q0Q1)) + r.z * (1 - 2 * Q12 - 2 * Q22) + t3
        #             s = DQClass.Quaternion(0, A, B, C)
        #             b = Base[n]
        #             u = s - b
        #             loss.append(u.norm_sq() - length_sq[n])
        #         while np.max(np.abs(loss)) > np.max(np.abs(lossVec)):
        #             descent_param *= 0.5
        #             X[3:] = X.To6Vec()[3:] - descent_param * addend[3:]  
        #             norm = np.sqrt(Q1**2 + Q2**2 + Q3**2)
        #             Q0 = np.sqrt((1+Q0)/2)  # cos(alpha/4)
        #             sinao4 = np.sqrt(1-cosao4**2)  # sin(alpha/4)
        #             X[:3] *= sinao4 / norm  # Normalize the rotation part
        #             Q1 = X[0]
        #             Q2 = X[1]
        #             Q3 = X[2]
        #             Q0Q1, Q0Q2, Q0Q3= Q0 * Q1, Q0 * Q2, Q0 * Q3
        #             Q1Q2, Q1Q3 = Q1 * Q2, Q2 * Q3
        #             Q2Q3 = Q2 * Q3
        #             Q02 = Q0 ** 2
        #             Q12 = Q1 ** 2
        #             Q22 = Q2 ** 2
        #             Q32 = Q3 ** 2
        #             t1 = X[3]
        #             t2 = X[4]
        #             t3 = X[5]
        #             loss = []
        #             for n in range(6):
        #                 r = TableID[n]
        #                 A = 2 * (r.y * (Q1Q2 - Q0Q3) + r.z * (Q1Q3 + Q0Q2)) + r.x * (1 - 2 * Q22 - 2 * Q32) + t1
        #                 B = 2 * (r.x * (Q1Q2 + Q0Q3) + r.z * (Q2Q3 - Q0Q1)) + r.y * (1 - 2 * Q12 - 2 * Q32) + t2
        #                 C = 2 * (r.x * (Q1Q3 - Q0Q2) + r.y * (Q2Q3 + Q0Q1)) + r.z * (1 - 2 * Q12 - 2 * Q22) + t3
        #                 s = DQClass.Quaternion(0, A, B, C)
        #                 b = Base[n]
        #                 u = s - b
        #                 loss.append(u.norm_sq() - length_sq[n])
        #             if descent_param < 1e-6:
        #                 break