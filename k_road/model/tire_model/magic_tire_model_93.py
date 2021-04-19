from math import *

# noinspection PyPep8Naming
from k_road.model.tire_model.lateral_tire_model import LateralTireModel


class MagicTireModel93(LateralTireModel):
    """
    see Bakker, E.; Pacejka, H. B.; Lidner,
        L. A new tire model with an application in vehicle dynamics studies. [S.l.], 1989
    see https://www.edy.es/dev/docs/pacejka-94-parameters-explained-a-comprehensive-guide/
    see https://andresmendes.github.io/Vehicle-Dynamics-Lateral/html/DocTirePacejka.html
    see https://www.cm-labs.com/vortexstudiodocumentation/Vortex_User_Documentation/Content/Concepts/tire_models.html
    see https://pdfs.semanticscholar.org/04aa/8fa25886103a97b92443346fc94bf4c6c50b.pdf
    see https://www.sciencedirect.com/science/article/pii/S1877705817319136
    see https://github.com/kktse/uwfm/tree/master/ymd/model
    see page 172 of Tyre and Vehicle Dynamics (Pacejka, 2002)
    see https://www.eng-tips.com/viewthread.cfm?qid=348016 for a source of params
    see http://www.racer.nl/reference/pacejka.htm
    """

    def __init__(
            self,
            R0: float = .313,  # wheel radius [m]
            Fz0: float = 4000,  # Nominal load [N] (aka FN0)
            m0: float = 9.3,  # tire mass [kg]
            V0: float = 16.67,  # nominal velocity [m/s]
            pCx1: float = 1.685,
            pDx1: float = 1.21,
            pDx2: float = -.037,
            pEx1: float = .344,
            pEx2: float = .095,
            pEx3: float = -.02,
            pEx4: float = 0.0,
            pKx1: float = 21.51,
            pKx2: float = -.163,
            pKx3: float = .245,
            pHx1: float = -.002,
            pHx2: float = .002,
            pVx1: float = 0.0,
            pVx2: float = 0.0,
            rBx1: float = 12.35,
            rBx2: float = -10.77,
            rCx1: float = 1.092,
            rHx1: float = .007,
            rEx1: float = 0.0,
            rEx2: float = 0.0,
            qsx1: float = 0.0,
            qsx2: float = 0.0,
            qsx3: float = 0.0,
            pCy1: float = 1.193,
            pDy1: float = -.99,
            pDy2: float = .145,
            pDy3: float = -11.23,
            pEy1: float = -1.003,
            pEy2: float = -.537,
            pEy3: float = -.083,
            pEy4: float = -4.787,
            pKy1: float = -14.95,
            pKy2: float = 2.130,
            pKy3: float = -.028,
            pHy1: float = .003,
            pHy2: float = -.001,
            pHy3: float = .075,
            pVy1: float = .045,
            pVy2: float = -.024,
            pVy3: float = -.532,
            pVy4: float = .039,
            rBy1: float = 6.461,
            rBy2: float = 4.196,
            rBy3: float = -.015,
            rCy1: float = 1.081,
            rEy1: float = 0.0,
            rEy2: float = 0.0,
            rHy1: float = .009,
            rHy2: float = 0.0,
            rVy1: float = .053,
            rVy2: float = -.073,
            rVy3: float = .517,
            rVy4: float = 35.44,
            rVy5: float = 1.9,
            rVy6: float = -10.71,
            qBz1: float = 8.964,
            qBz2: float = -1.106,
            qBz3: float = -.842,
            qBz4: float = 0.0,
            qBz5: float = -.227,
            qBz9: float = 18.47,
            qBz10: float = 0.0,
            qCz1: float = 1.18,
            qDz1: float = .1,
            qDz2: float = -.001,
            qDz3: float = .007,
            qDz4: float = 13.05,
            qDz6: float = -.008,
            qDz7: float = 0.0,
            qDz8: float = -.296,
            qDz9: float = -.009,
            qEz1: float = -1.609,
            qEz2: float = -.359,
            qEz3: float = 0.0,
            qEz4: float = .174,
            qEz5: float = -.896,
            qHz1: float = .007,
            qHz2: float = -.002,
            qHz3: float = .147,
            qHz4: float = .004,
            ssz1: float = .043,
            ssz2: float = .001,
            ssz3: float = .731,
            ssz4: float = -.238,
            q1ay: float = .109,
            qma: float = .237,
            qcbx0z: float = 121.4,
            qkbxz: float = .228,
            qcvt0: float = 61.96,
            q1axz: float = .071,
            qmb: float = .763,
            qcby: float = 40.05,
            qkby: float = .284,
            qcbgp: float = 20.33,
            qIby: float = .696,
            qmc: float = .108,
            qccx: float = 391.9,
            qkcx: float = .91,
            qccp: float = 55.82,
            qIbxz: float = .357,
            qccy: float = 62.7,
            qkcy: float = .91,
            qkbt: float = .08,
            qIc: float = .055,
            qkbgp: float = .038,
            qkcp: float = .834,
            qV1: float = 7.1e-5,
            qV2: float = 2.489,
            qFz1: float = 13.37,
            qFz2: float = 14.35,
            qsy1: float = .01,
            qsy2: float = 0.0,
            qsy3: float = 0.0,
            qsy4: float = 0.0,
            qa1: float = .135,
            qa2: float = .035,
            qbvxz: float = 3.957,
            qbvt: float = 3.957,
            Breff: float = 9.0,
            Dreff: float = .23,
            Freff: float = .01,
            qFcx1: float = .1,
            qFcy1: float = .3,
            qFcx2: float = 0.0,
            qFcy2: float = 0.0,
            lmV: float = 0.0,
            lFz0: float = 1.0,  # pure slip scaling factor: nominal (rated) load
            lmx: float = 1.0,  # pure slip scaling factor: peak friction coefficient
            lmy: float = 1.0,  # pure slip scaling factor: peak friction coefficient
            luV: float = 1.0,  # pure slip scaling factor: with slip speed Vs decaying friction
            lKxk: float = 1.0,  # pure slip scaling factor: brake slip stiffness
            lKya: float = 1.0,  # pure slip scaling factor: cornering stiffness
            lCx: float = 1.0,  # pure slip scaling factor: shape factor
            lEx: float = 1.0,  # pure slip scaling factor: curvature factor
            lHx: float = 1.0,  # pure slip scaling factor: horizontal shift
            lVx: float = 1.0,  # pure slip scaling factor: vertical shift
            lCy: float = 1.0,  # pure slip scaling factor: shape factor
            lEy: float = 1.0,  # pure slip scaling factor: curvature factor
            lHy: float = 1.0,  # pure slip scaling factor: horizontal shift
            lVy: float = 1.0,  # pure slip scaling factor: vertical shift
            lKyg: float = 1.0,  # pure slip scaling factor: camber force stiffness
            lKzg: float = 1.0,  # pure slip scaling factor: camber torque stiffness
            lt: float = 1.0,  # pure slip scaling factor: pneumatic trail (effecting aligning torque stiffness)
            lMr: float = 1.0,  # pure slip scaling factor: residual torque
            lxa: float = 1.0,  # combined slip  scaling factor: a influence on Fx(k)
            lyk: float = 1.0,  # combined slip  scaling factor: k influence on Fy(a)
            lVyk: float = 1.0,  # combined slip  scaling factor: k induced 'ply-steer' Fy
            ls: float = 1.0,  # combined slip  scaling factor: Mz moment arm of Fx
            lCz: float = 1.0,  # other scaling factor: radial tyre stiffness
            lMx: float = 1.0,  # other scaling factor: overturning couple stiffness
            lMy: float = 1.0,  # other scaling factor: rolling resistance moment
            lKyy: float = 1.0,
            Am: float = 10,
            evx: float = 1e-6,
            ex: float = 1e-6,
            ey: float = 1e-6,
            evc: float = 1e-6,
            ek: float = 1e-6,
            z0: float = 1.0,
            z1: float = 1.0,
            z2: float = 1.0,
            z3: float = 1.0,
            z4: float = 1.0,
            z5: float = 1.0,
            z6: float = 1.0,
            z7: float = 1.0,
            z8: float = 1.0,
    ):
        self.R0: float = R0
        self.Fz0: float = Fz0
        self.m0: float = m0
        self.V0: float = V0
        self.pCx1: float = pCx1
        self.pDx1: float = pDx1
        self.pDx2: float = pDx2
        self.pEx1: float = pEx1
        self.pEx2: float = pEx2
        self.pEx3: float = pEx3
        self.pEx4: float = pEx4
        self.pKx1: float = pKx1
        self.pKx2: float = pKx2
        self.pKx3: float = pKx3
        self.pHx1: float = pHx1
        self.pHx2: float = pHx2
        self.pVx1: float = pVx1
        self.pVx2: float = pVx2
        self.rBx1: float = rBx1
        self.rBx2: float = rBx2
        self.rCx1: float = rCx1
        self.rHx1: float = rHx1
        self.rEx1: float = rEx1
        self.rEx2: float = rEx2
        self.qsx1: float = qsx1
        self.qsx2: float = qsx2
        self.qsx3: float = qsx3
        self.pCy1: float = pCy1
        self.pDy1: float = pDy1
        self.pDy2: float = pDy2
        self.pDy3: float = pDy3
        self.pEy1: float = pEy1
        self.pEy2: float = pEy2
        self.pEy3: float = pEy3
        self.pEy4: float = pEy4
        self.pKy1: float = pKy1
        self.pKy2: float = pKy2
        self.pKy3: float = pKy3
        self.pHy1: float = pHy1
        self.pHy2: float = pHy2
        self.pHy3: float = pHy3
        self.pVy1: float = pVy1
        self.pVy2: float = pVy2
        self.pVy3: float = pVy3
        self.pVy4: float = pVy4
        self.rBy1: float = rBy1
        self.rBy2: float = rBy2
        self.rBy3: float = rBy3
        self.rCy1: float = rCy1
        self.rEy1: float = rEy1
        self.rEy2: float = rEy2
        self.rHy1: float = rHy1
        self.rHy2: float = rHy2
        self.rVy1: float = rVy1
        self.rVy2: float = rVy2
        self.rVy3: float = rVy3
        self.rVy4: float = rVy4
        self.rVy5: float = rVy5
        self.rVy6: float = rVy6
        self.qBz1: float = qBz1
        self.qBz2: float = qBz2
        self.qBz3: float = qBz3
        self.qBz4: float = qBz4
        self.qBz5: float = qBz5
        self.qBz9: float = qBz9
        self.qBz10: float = qBz10
        self.qCz1: float = qCz1
        self.qDz1: float = qDz1
        self.qDz2: float = qDz2
        self.qDz3: float = qDz3
        self.qDz4: float = qDz4
        self.qDz6: float = qDz6
        self.qDz7: float = qDz7
        self.qDz8: float = qDz8
        self.qDz9: float = qDz9
        self.qEz1: float = qEz1
        self.qEz2: float = qEz2
        self.qEz3: float = qEz3
        self.qEz4: float = qEz4
        self.qEz5: float = qEz5
        self.qHz1: float = qHz1
        self.qHz2: float = qHz2
        self.qHz3: float = qHz3
        self.qHz4: float = qHz4
        self.ssz1: float = ssz1
        self.ssz2: float = ssz2
        self.ssz3: float = ssz3
        self.ssz4: float = ssz4
        self.q1ay: float = q1ay
        self.qma: float = qma
        self.qcbx0z: float = qcbx0z
        self.qkbxz: float = qkbxz
        self.qcvt0: float = qcvt0
        self.q1axz: float = q1axz
        self.qmb: float = qmb
        self.qcby: float = qcby
        self.qkby: float = qkby
        self.qcbgp: float = qcbgp
        self.qIby: float = qIby
        self.qmc: float = qmc
        self.qccx: float = qccx
        self.qkcx: float = qkcx
        self.qccp: float = qccp
        self.qIbxz: float = qIbxz
        self.qccy: float = qccy
        self.qkcy: float = qkcy
        self.qkbt: float = qkbt
        self.qIc: float = qIc
        self.qkbgp: float = qkbgp
        self.qkcp: float = qkcp
        self.qV1: float = qV1
        self.qV2: float = qV2
        self.qFz1: float = qFz1
        self.qFz2: float = qFz2
        self.qsy1: float = qsy1
        self.qsy2: float = qsy2
        self.qsy3: float = qsy3
        self.qsy4: float = qsy4
        self.qa1: float = qa1
        self.qa2: float = qa2
        self.qbvxz: float = qbvxz
        self.qbvt: float = qbvt
        self.Breff: float = Breff
        self.Dreff: float = Dreff
        self.Freff: float = Freff
        self.qFcx1: float = qFcx1
        self.qFcy1: float = qFcy1
        self.qFcx2: float = qFcx2
        self.qFcy2: float = qFcy2
        self.lmV: float = lmV
        self.lFz0: float = lFz0
        self.lmx: float = lmx
        self.lmy: float = lmy
        self.luV: float = luV
        self.lKxk: float = lKxk
        self.lKya: float = lKya
        self.lCx: float = lCx
        self.lEx: float = lEx
        self.lHx: float = lHx
        self.lVx: float = lVx
        self.lCy: float = lCy
        self.lEy: float = lEy
        self.lHy: float = lHy
        self.lVy: float = lVy
        self.lKyg: float = lKyg
        self.lKzg: float = lKzg
        self.lt: float = lt
        self.lMr: float = lMr
        self.lxa: float = lxa
        self.lyk: float = lyk
        self.lVyk: float = lVyk
        self.ls: float = ls
        self.lCz: float = lCz
        self.lMx: float = lMx
        self.lMy: float = lMy
        self.lKyy: float = lKyy
        self.Am: float = Am
        self.evx: float = evx
        self.ex: float = ex
        self.ey: float = ey
        self.evc: float = evc
        self.ek: float = ek
        self.z0: float = z0
        self.z1: float = z1
        self.z2: float = z2
        self.z3: float = z3
        self.z4: float = z4
        self.z5: float = z5
        self.z6: float = z6
        self.z7: float = z7
        self.z8: float = z8

    def estimate_stiffness_from_mass_and_spacing(self):
        return self.pCy1 * self.lCy  # eqn 4.E21: (>0)

    def get_lateral_force(
            self,
            Fz: float,  # Fz = vertical force on tire (load)
            a: float,  # slip angle
            Fx0: float,  # longitudinal force [N]
            Vc: float,  # wheel contact center velocity magnitude
            k: float,  # longitudinal slip
            g: float = 0.0,  # g (gamma) = camber angle
    ) -> (float, float):
        Fz /= 1000.0  # convert N to kN

        Vcx = Vc * cos(a)
        Vcy = Vc * sin(a)
        a_prime = a  # what does a' mean?

        absVcx = abs(Vcx) + self.evx  # as suggested on p 185 to avoid singularity
        a_star = - Vcy / absVcx  # eqn 4.E3: adjusted slip angle ('lateral slip')
        Vsx = -k * Vcx  # longitudinal slip velocity
        Vs = Vsx  # not sure if this is right
        Fz0p = self.lFz0 * self.Fz0  # eqn 4.E1 Adapted nominal load
        dfz = (Fz - Fz0p) / Fz0p  # eqn 4.E2 Normalized change in vertical load
        g_star = sin(g)  # eqn 4.E4: spin due to camber angle

        # Lateral Force (pure side slip) ----------------
        # inputs: Vs, a_prime, g (gamma)

        Cy = self.pCy1 * self.lCy  # eqn 4.E21: (>0)

        my = (self.pDy1 + self.pDy2 * dfz) * (1.0 - self.pDy3 * g_star ** 2) * self.lmy / (
                1.0 + self.lmV * Vs / self.V0)  # eqn 4.E23: (>0)
        Dy = my * Fz * self.z2  # eqn 4.E22

        Fz0_prime = self.lFz0 * self.Fz0  # eqn 4.E1: effect of tyre with different nominal load
        Kya0 = self.pKy1 * Fz0_prime * sin(2.0 * atan(Fz / (self.pKy2 * Fz0_prime))) * self.lKya  # eqn 4.E25
        Kya = Kya0 * (1.0 - self.pKy3 * g_star ** 2) * self.z3  # eqn 4.E26
        By = Kya / (Cy * Dy + self.ey)  # eqn 4.E27
        SHy = (
                      self.pHy1 + self.pHy2 * dfz) * self.lHy + self.pHy3 * g_star * self.lKyy * self.z0 + \
              self.z4 - 1.0  # eqn 4.E28
        ay = a_prime + SHy
        Ey = (self.pEy1 + self.pEy2 * dfz) * (
                1.0 - (self.pEy3 + self.pEy4 * g_star) * copysign(1.0, ay)) * self.lEy  # eqn 4.E24

        By_ay = By * ay
        Fy0 = Dy * sin(Cy * atan(By_ay - Ey * (By_ay - atan(By_ay))))  # eqn 4.E19: Lateral Force (pure side slip)

        # Lateral Force (combined slip) ----------------
        # inputs: a_star, a_prime, k, Fy0
        Byk = self.rBy1 * cos(atan(self.rBy2 * (a_prime - self.rBy3))) * self.lyk  # eqn 4.E62: (>0)
        Cyk = self.rCy1  # eqn 4.E63
        Eyk = self.rEy1 + self.rEy2 * dfz  # eqn 4.E64: (<=1)
        SHyk = self.rHy1 + self.rHy2 * dfz  # eqn 4.E65
        DVyk = my * Fz * (self.rVy1 + self.rVy2 * dfz + self.rVy3 * g_star) * cos(
            atan(self.rVy4 * a_star)) * self.z2  # eqn 4.E67
        SVyk = DVyk * sin(self.rVy5 * atan(self.rVy6 * k))  # eqn 4.E66

        kS = k + SHyk  # eqn 4.E61

        Byk_SHyk = Byk * SHyk
        Gyk0 = cos(Cyk * atan(Byk_SHyk - Eyk * (Byk_SHyk - atan(Byk_SHyk))))  # eqn 4.E60

        Byk_kS = Byk * kS
        Gyk = cos(Cyk * atan(Byk_kS - Eyk * (Byk_kS - atan(Byk_kS)))) / Gyk0  # eqn 4.E59: (>0)

        Fy = Gyk * Fy0 + SVyk  # eqn 4.E58: Lateral Force (combined slip)

        return Fy, Fy0

    def calc_all_values(
            self,
            Fz,  # Fz = vertical force on tire (load)
            Vc,  # wheel contact center velocity magnitude
            a,  # slip angle
            k,  # longitudinal slip
            g,  # g (gamma) = camber angle
    ):
        Fz /= 1000.0  # convert N to kN

        # p. 188
        # k = -Vsx / absVcx  # longitudinal slip

        # a = atan2(Vcy, Vcx)
        Vcx = Vc * cos(a)
        Vcy = Vc * sin(a)
        a_prime = a  # what does a' mean?

        absVcx = abs(Vcx) + self.evx  # as suggested on p 185 to avoid singularity
        a_star = - Vcy / absVcx  # eqn 4.E3: adjusted slip angle ('lateral slip')
        # a_star = tan(a) * copysign(1, Vcx) # eqn 4.E3: adjusted slip angle ('lateral slip') alternate formulation
        Vsx = -k * Vcx  # longitudinal slip velocity
        Vs = Vsx  # not sure if this is right
        Fz0p = self.lFz0 * self.Fz0  # eqn 4.E1 Adapted nominal load
        dfz = (Fz - Fz0p) / Fz0p  # eqn 4.E2 Normalized change in vertical load
        g_star = sin(g)  # eqn 4.E4: spin due to camber angle

        # Longitudinal Force (pure longitudinal slip) ---------------
        # inputs: k, kx, Vs

        SHx = (self.pHx1 + self.pHx2 * dfz) * self.lHx  # eqn 4.E17
        kx = k + SHx

        lmxp = self.Am * self.lmx / (
                1.0 + (self.Am - 1.0) * self.lmx)  # eqn 4.E8 : digressive friction factor (x-component)
        Svx = Fz * (self.pVx1 + self.pVx2 * dfz) * (abs(Vcx) / absVcx) * self.lVx * lmxp * self.z1  # eqn 4.E18
        Kxk = Fz * (self.pKx1 + self.pKx2 * dfz) * exp(self.pKx3 * dfz) * self.lKxk  # eqn 4.E15
        Ex = (self.pEx1 + self.pEx2 * dfz + self.pEx3 * dfz ** 2) * (
                1.0 - self.pEx4 * copysign(1.0, kx)) * self.lEx  # eqn 4.E14
        mx = (self.pDx1 + self.pDx2 * dfz) * self.lmx / (1.0 + self.luV * Vs / self.V0)  # eqn 4.E13
        Dx = mx * Fz * self.z1  # eqn 4.E12
        Cx = self.pCx1 * self.lCx  # eqn 4.E11
        Bx = Kxk / (Cx * Dx + self.ex)  # eqn 4.E16

        Bx_kx = Bx * kx
        Fx0 = Dx * sin(Cx * atan(
            Bx_kx - Ex * (Bx_kx - atan(Bx_kx)))) + Svx  # eqn 4.E9: Longitudinal Force (pure longitudinal slip)

        # Longitudinal Force (combined slip) ----------------
        # inputs: a_star, k, Fx0
        SHxa = self.rHx1  # 4.E57
        Exa = self.rEx1 + self.rEx2 * dfz  # 4.E56: (<= 1)
        Cxa = self.rCx1  # eqn 4.E55
        Bxa = self.rBx1 * cos(atan(self.rBx2 * k)) * self.lxa  # eqn 4.E54
        aS = a_star + SHxa  # eqn 4.E53

        Bxa_SHxa = Bxa * SHxa
        Gxao = cos(Cxa * atan(Bxa_SHxa - Exa * (Bxa_SHxa - atan(Bxa_SHxa))))  # eqn 4.E42

        Bxa_aS = Bxa * aS
        Gxa = cos(Cxa * atan(Bxa_aS - Exa * (Bxa_aS - atan(Bxa_aS)))) / Gxao  # eqn 4.E41: (>0)

        Fx = Gxa * Fx0  # eqn 4.E50: Longitudinal Force (combined slip)

        # Lateral Force (pure side slip) ----------------
        # inputs: Vs, a_prime, g (gamma)

        Cy = self.pCy1 * self.lCy  # eqn 4.E21: (>0)

        my = (self.pDy1 + self.pDy2 * dfz) * (1.0 - self.pDy3 * g_star ** 2) * self.lmy / (
                1.0 + self.lmV * Vs / self.V0)  # eqn 4.E23: (>0)
        Dy = my * Fz * self.z2  # eqn 4.E22

        Fz0_prime = self.lFz0 * self.Fz0  # eqn 4.E1: effect of tyre with different nominal load
        Kya0 = self.pKy1 * Fz0_prime * sin(2.0 * atan(Fz / (self.pKy2 * Fz0_prime))) * self.lKya  # eqn 4.E25
        Kya = Kya0 * (1.0 - self.pKy3 * g_star ** 2) * self.z3  # eqn 4.E26
        By = Kya / (Cy * Dy + self.ey)  # eqn 4.E27
        SHy = (
                      self.pHy1 + self.pHy2 * dfz) * self.lHy + self.pHy3 * g_star * self.lKyy * self.z0 + \
              self.z4 - 1.0  # eqn 4.E28
        ay = a_prime + SHy
        Ey = (self.pEy1 + self.pEy2 * dfz) * (
                1.0 - (self.pEy3 + self.pEy4 * g_star) * copysign(1.0, ay)) * self.lEy  # eqn 4.E24
        lmyp = self.Am * self.lmy / (
                1.0 + (self.Am - 1.0) * self.lmy)  # eqn 4.E8 : digressive friction factor (y-component)
        SVy = Fz * ((self.pVy1 + self.pVy2 * dfz) * self.lVy + (
                self.pVy3 + self.pVy4 * dfz) * g_star * self.lKyg) * lmyp * self.z2  # eqn 4.E29
        Kyg0 = self.lKyy * (self.pHy3 * Kya0 + Fz * (self.pVy3 + self.pVy4 * dfz))  # eqn 4.E30

        By_ay = By * ay
        Fy0 = Dy * sin(Cy * atan(By_ay - Ey * (By_ay - atan(By_ay))))  # eqn 4.E19: Lateral Force (pure side slip)

        # Lateral Force (combined slip) ----------------
        # inputs: a_star, a_prime, k, Fy0
        Byk = self.rBy1 * cos(atan(self.rBy2 * (a_prime - self.rBy3))) * self.lyk  # eqn 4.E62: (>0)
        Cyk = self.rCy1  # eqn 4.E63
        Eyk = self.rEy1 + self.rEy2 * dfz  # eqn 4.E64: (<=1)
        SHyk = self.rHy1 + self.rHy2 * dfz  # eqn 4.E65
        DVyk = my * Fz * (self.rVy1 + self.rVy2 * dfz + self.rVy3 * g_star) * cos(
            atan(self.rVy4 * a_star)) * self.z2  # eqn 4.E67
        SVyk = DVyk * sin(self.rVy5 * atan(self.rVy6 * k))  # eqn 4.E66

        kS = k + SHyk  # eqn 4.E61

        Byk_SHyk = Byk * SHyk
        Gyk0 = cos(Cyk * atan(Byk_SHyk - Eyk * (Byk_SHyk - atan(Byk_SHyk))))  # eqn 4.E60

        Byk_kS = Byk * kS
        Gyk = cos(Cyk * atan(Byk_kS - Eyk * (Byk_kS - atan(Byk_kS)))) / Gyk0  # eqn 4.E59: (>0)

        Fy = Gyk * Fy0 + SVyk  # eqn 4.E58: Lateral Force (combined slip)

        # Aligning Torque (pure side slip) ------------------
        # inputs:
        Vc_prime = Vc + self.evc  # eqn 4.E7
        cos_prime_a = Vcx / Vc_prime  # eqn 4.E6

        Dt0 = Fz * (self.R0 / Fz0_prime) * (self.qDz1 + self.qDz2 * dfz) * self.lt * copysign(1, Vcx)  # eqn 4.E42
        Dt = Dt0 * (1 + self.qDz3 + g_star + self.qDz4 * g_star ** 2) * self.z5  # eqn 4.E43

        Ct = self.qCz1  # eqn 4.E41: (>0)
        Bt = (self.qBz1 + self.qBz2 * dfz + self.qBz3 * dfz ** 2) * (
                1 + self.qBz4 * g_star + self.qBz5 * abs(g_star)) * self.lKya / self.lmy  # eqn 4.E40

        Kya_prime = Kya + self.ek  # eqn 4.E39
        SHf = SHy + SVy / Kya_prime  # eqn 4.E38
        ar = a_prime * SHf  # eqn 4.E37: (=af)

        Br = self.qBz10 * By * Cy  # eqn 4.E45
        Cr = self.z7  # eqn 4.E46
        Dr = Fz * self.R0 * ((self.qDz6 + self.qDz7 * dfz) * self.lMr * self.z2 +
                             (self.qDz8 + self.qDz9 * dfz) * g_star * self.lKzg * self.z0) \
             * cos_prime_a * self.lmy * copysign(1, Vcx) + self.z8 - 1  # eqn 4.E47
        Kza0 = Dt0 * Kya0  # eqn 4.E48
        Kzg0 = Fz * self.R0 * (self.qDz8 + self.qDz9 * dfz) * self.lKzg - Dt0 * Kyg0  # eqn 4.E49

        Mzr0 = Dr * cos(Cr * atan(Br * ar))  # eqn 4.E36
        SHt = self.qHz1 + self.qHz2 * dfz + (self.qHz3 + self.qHz4 * dfz) * g_star  # eqn 4.E35
        at = a_prime + SHt  # eqn 4.E34

        Et = (self.qEz1 + self.qEz2 * dfz + self.qEz3 * dfz ** 2) * (
                1 + (self.qEz4 + self.qEz5 * g_star) * (2 / pi) * atan(Bt * Ct * at))  # eqn 4.E44: (<=1)

        Bt_at = Bt * at
        to = Dt * cos(Ct * atan(Bt_at - Et * (Bt_at - atan(Bt_at)))) * cos_prime_a  # eqn 4.E33
        Mz0_prime = -to * Fy0  # eqn 4.E32
        Mz0 = Mz0_prime + Mzr0  # eqn 4.E31: Aligning Torque (pure side slip)

        # Overturning Couple - rotates around x axis
        Mx = Fz * self.R0 * \
             (self.qsx1 - self.qsx2 * g_star + self.qsx3 * Fy / Fz0_prime) * self.lMx  # eqn 4.E69: Overturning Couple

        # Rolling Resistance Moment -- rotates around y axis
        Vr = Vcx - Vsx  # wheel linear speed of rolling
        My = -Fz * self.R0 * (self.qsy1 *
                              atan(
                                  Vr / self.V0) + self.qsy2 * Fx / Fz0_prime) * self.lMy  # eqn 4.E70: Rolling
        # Resistance Moment

        # Aligning Torque (combined slip) -- rotates around z axis

        Fy_prime = Fy - SVyk  # eqn 4.E74
        ateq = sqrt(at ** 2 + (Kxk / Kya_prime) ** 2 * k ** 2) * copysign(1, at)
        areq = sqrt(ar ** 2 + (Kxk / Kya_prime) ** 2 * k ** 2) * copysign(1, ar)
        s = self.R0 * (self.ssz1 + self.ssz2 * (Fy / Fz0_prime) + (
                self.ssz3 + self.ssz4 * dfz) * g_star) * self.ls  # eqn 4.E76
        Mzr = Dr * cos(Cr * atan(Br * areq))  # eqn 4.E75
        Bt_ateq = Bt * ateq
        t = Dt * cos(Ct * atan(Bt_ateq - Et * (Bt_ateq - atan(Bt_ateq)))) * cos_prime_a  # eqn 4.E73
        Mz_prime = -t * Fy_prime  # eqn 4.E72
        Mz = Mz_prime + Mzr + s * Fx  # eqn 4.E71: Aligning Torque (combined slip)

        return Fx, Fy, Fx0, Fy0, Mx, My, Mz, Mz0
