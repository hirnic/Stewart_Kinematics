# This is a program that carries out computations for kinematics using dual quaternions.
import math
import numpy as np


class Quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"

    def __add__(self, other):
        return Quaternion(self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Quaternion(self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, (int, float, np.float32)):
            return Quaternion(self.w * other, self.x * other, self.y * other, self.z * other)
        else:
            w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
            x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
            y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
            z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
            return Quaternion(w, x, y, z)

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return math.sqrt(self.w**2 + self.x**2 + self.y**2+self.z**2)

    def norm_sq(self):
        return self.w**2 + self.x**2 + self.y**2+self.z**2

    def absmax(self):
        return max(abs(self.w), abs(self.x), abs(self.y), abs(self.z))

    def inverse(self):
        conjugate = self.conjugate()
        norm = self.norm()**2
        try:
            return Quaternion(conjugate.w / norm, conjugate.x / norm, conjugate.y / norm, conjugate.z / norm)
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def __truediv__(self, other):
        return self * other.inverse()

    def normalization(self):
        norm = self.norm()
        try:
            return Quaternion(self.w / norm, self.x / norm, self.y / norm, self.z / norm)
        except ZeroDivisionError:
            return Quaternion(0, 0, 0, 0)

    def ImaginaryPart(self):
        return Quaternion(0, self.x, self.y, self.z)

    def RealPart(self):
        return Quaternion(self.w, 0, 0, 0)

    def DotProduct(self, other):
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def ParallelPart(self, other):  # gives the projection of self onto other (component of self parallel to other)
        d = other.normalization()
        return d * d.DotProduct(self)

    def OrthogonalPart(self, other):  #give the projection of self onto orthogonal component of other (component of self orthogonal to other)
        return self - self.ParallelPart(other)

    def Exponential(self):
        a = self.ImaginaryPart()
        m = a.norm()
        r = self.w
        C = Quaternion(math.cos(m), 0, 0, 0)
        S = Quaternion(math.sin(m), 0, 0, 0)
        M = Quaternion(math.exp(r), 0, 0, 0)
        return M * (C + S * a)

    def Logarithm(self):
        N = self.norm()
        Q = self.normalization()
        Re = Q.w
        Im = Q.ImaginaryPart().normalization()
        return Quaternion(math.log(N), 0, 0, 0) + Quaternion(math.atan2(N, Re), 0, 0, 0) * Im

    def ToFullVec(self):
        return [self.w, self.x, self.y, self.z]

    def ToPureVec(self):
        return [self.x, self.y, self.z]

    def PoseIt(self, other):  # Other must be a pose.
        return other.Q * self * other.Q.conjugate()  + other.T

    def Rotate(self, other): # Self is the vector being rotated, other is the rotation quaternion
        return other * self * other.conjugate()


class DQuaternion:
    def __init__(self, A, B):
        self.A = A
        self.B = B

    def __repr__(self):
        return f"DQuaternion({self.A}, {self.B})"

    def __add__(self, other):
        return DQuaternion(self.A + other.A, self.B + other.B)

    def __sub__(self, other):
        return DQuaternion(self.A - other.A, self.B - other.B)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DQuaternion(self.A * other, self.B * other)
        return DQuaternion(self.A * other.A, self.A * other.B + self.B * other.A)

    def copy(self):
        return DQuaternion(self.A, self.B)

    def conjugate(self):
        return DQuaternion(self.A.conjugate(), self.B.conjugate())

    def overbar(self):
        return DQuaternion(self.A, Quaternion(-1, 0, 0, 0) * self.B)

    def inverse(self):
        try:
            return DQuaternion(self.A.inverse(), self.A.inverse() * self.B * self.A.inverse()).overbar()
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def __truediv__(self, other):
        return self * other.inverse()

    def norm(self):
        try:
            R = self.A.norm()
            D = self.A.DotProduct(self.B)/self.A.norm()
            if self.A == Quaternion(0, 0, 0, 0):
                return DQuaternion(Quaternion(0, 0, 0, 0,), Quaternion(0, 0, 0, 0))
            else:
                return DQuaternion(Quaternion(R, 0, 0, 0), Quaternion(D, 0, 0, 0))
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def normalization(self):
        try:
            R = self.A.norm()
            D1 = self.B * (1/R)
            D2 =  self.A * self.A.DotProduct(self.B) * (1/R**3)
            return DQuaternion(self.A * (1/R), D1 - D2)
        except ZeroDivisionError:
            return DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0))

    def Exponential(self, t = 2):
        import numpy as np
        c = self.A.w
        d = self.A.ImaginaryPart()
        x = self.B.w
        y = self.B.ImaginaryPart()
        yPar = y.ParallelPart(d)
        yPerp = y.OrthogonalPart(d)
        v1 = yPar.norm()
        v2 = yPerp.norm()
        a = yPar.normalization()
        b = yPerp.normalization()
        w = d.norm()
        scale = DQuaternion(Quaternion(1, 0, 0, 0), Quaternion(x , 0, 0, 0)) * math.exp(c)
        if v1+v2 == 0:
            return scale * DQuaternion(d.Exponential(), Quaternion(0, 0, 0, 0))
        if w == 0:
            return scale * DQuaternion(Quaternion(1, 0, 0, 0), y)
        half_tw = 1/2 * t * w
        cos = math.cos(half_tw)
        sin = math.sin(half_tw)
        non_dual_part = Quaternion(cos, 0, 0, 0) + a * sin
        dual_part =  non_dual_part * a * (1/2 * v1 * t) + b * (v2/w * sin)
        return scale * DQuaternion(non_dual_part, dual_part)

    def Logarithm(self):
        # Note, this code assumes that the dual quaternion is normalized.
        NDPart = self.A
        DPart = self.B
        c = NDPart.w
        s = NDPart.ImaginaryPart().norm()
        a = NDPart.ImaginaryPart().normalization()
        t = math.atan2(s, c)
        x = DPart.w
        y = DPart.ImaginaryPart()
        yPar = y.ParallelPart(a)
        yPerp = y.OrthogonalPart(a)
        b = yPerp.normalization()
        y1 = yPar.norm()
        y2 = yPerp.norm()
        tysb = 0
        if s == 0:
            tysb = Quaternion(0, 0, 0, 0)
        else:
            tysb = b * (t * y2 / s)
        non_dual_part = a * t
        dual_part =  (a*(c*y1 - s*x)) + tysb
        return DQuaternion(non_dual_part, dual_part)

    def slerp(self, other, t):
        T = DQuaternion(Quaternion(t, 0, 0, 0), Quaternion(0, 0, 0, 0))
        try:
            A = self.conjugate() * other
            L = T * A.Logarithm()
            return self * L.Exponential()
        except ZeroDivisionError:
            print("Cannot invert zero divisors!")

    def ToFullVec(self):
        return [self.A.w, self.A.x, self.A.y, self.A.z, self.B.w, self.B.x, self.B.y, self.B.z]

    def To6Vec(self):
        return [self.A.x, self.A.y, self.A.z, self.B.x, self.B.y, self.B.z]

    def size(self):
        return math.sqrt(self.A.norm()**2 + self.B.norm()**2)

    def absmax(self):
        return max(self.A.absmax(), self.B.absmax())


def IdentityQ():
    return Quaternion(1, 0, 0, 0)


def IdentityDQ():
    return DQuaternion(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0))


def BasisDQ():
    X = [DQuaternion(Quaternion(1, 0, 0, 0), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 1, 0, 0), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 1, 0), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 1), Quaternion(0, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(1, 0, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 1, 0, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 1, 0)),
         DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 1))]
    return X


def ZeroDQ():
    return DQuaternion(Quaternion(0, 0, 0, 0), Quaternion(0, 0, 0, 0))


def ToQuaternion(x):  # x must be a 3-vector
    if len(x) != 4:
        print("You cannot turn this vector into a rotation quaternion. The input must be a 1D array of length 4.")
    else:
        return Quaternion(x[0], x[1], x[2], x[3])


def ToVectorQuaternion(x):  # x must be a 3-vector
    if len(x) != 3:
        print("You cannot turn this vector into a rotation quaternion. The input must be a 1D array of length 3.")
    else:
        return Quaternion(0, x[0], x[1], x[2])

def ToQuaternionRotation(x):  # x must be a 3-vector of norm < 1
    if len(x) != 3:
        print("You cannot turn this vector into a rotation quaternion. The input must be a 1D array of length 3.")
        return
    S = Quaternion(0, x[0], x[1], x[2])
    S_sq =1 - S.x**2 - S.y**2 - S.z**2
    if S_sq < 0:
        #print("Rotation vectors must have norm less than or equal to 1!")
        return
    else:
        try:
            C = math.sqrt(S_sq)
        except:
            C = 0
        return Quaternion(C, 0, 0, 0) + S

def ToQuaternionTranslation(x):  # x must be a 3-vector
    if len(x) != 3:
        print("You cannot turn this vector into a translation quaternion. The input must be a 1D array of length 3.")
    else:
        return Quaternion(0, x[0], x[1], x[2])


def ToDualQuaternion(x):
    if len(x) != 8:
        print("You cannot turn this vector into a dual quaternion. Input must be a 1D array of length 8.")
    else:
        Q = ToQuaternion(x[:4])
        t = ToQuaternion(x[4:])
        return DQuaternion(Q, t)


def ToVectorDualQuaternion(x):
    if len(x) != 6:
        print("You cannot turn this vector into a unit dual quaternion. Input must be a 1D array of length 6.")
    else:
        Q = ToVectorQuaternion(x[:3])
        t = ToQuaternionTranslation(x[3:])
        return DQuaternion(Q, t)


def ToPoseDualQuaternion(x):
    if len(x) != 6:
        print("You cannot turn this vector into a unit dual quaternion. Input must be a 1D array of length 6.")
    else:
        Q = ToQuaternionRotation(x[:3])
        t = ToQuaternionTranslation(x[3:])
        B = t * Q * (1 / 2)
        return DQuaternion(Q, B)


class Pose:
    def __init__(self, Q, T):
        self.Q = Q
        self.T = T

    def __repr__(self):
        return f"Pose({self.Q}, {self.T})"

def ToPose(x):
    Q = ToQuaternionRotation(x[:3])
    T = ToQuaternionTranslation(x[3:])
    return Pose(Q,T)