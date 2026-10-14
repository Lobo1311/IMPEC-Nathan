import numpy as np


ECO2, EH2O, ENaCl = range(3)

def debug_stop(message="Numerical validation failed"):
    raise RuntimeError(message)

def norm(x):
    return np.linalg.norm(x)

def _print_errors(label, errors):
    errors = np.asarray(errors)
    print(f"{label} derivative errors = {errors}")

    slopes = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0:
            slope = (
                np.log(errors[i + 1]) - np.log(errors[i])
            ) / (
                np.log(i + 2.0) - np.log(i + 1.0)
            )
        else:
            slope = np.nan
        slopes.append(slope)

    print(" ".join(str(v) for v in slopes))

#* -------------------------------------------------------------------

class Component:
    def __init__(self):
        self.fK = 0.0
        self.fK0 = 0.0
        self.fDK = np.zeros(2)
        self.fDK0 = np.zeros(2)
        self.fIsInverse = False

        self.fx = np.zeros(2)
        self.fDx = np.zeros((2, 2))

        self.fz = 0.0
        self.fDz = np.zeros(2)
        self.fRefVals = np.zeros(2)

    def SetK(self, K, GradK, InverseK):
        self.fIsInverse = InverseK
        self.fK = K
        self.fK0 = K
        self.fDK = np.array(GradK, dtype=float).copy()
        self.fDK0 = np.array(GradK, dtype=float).copy()

    def SetReferenceValues(self, RefVals):
        self.fRefVals = np.array(RefVals, dtype=float).copy()

    def Print(self, name):
        print(f"Molar fraction of {name}: {self.fx}")
        print(f"Dx:\n{self.fDx}")
        print(f"K fugacity ratio of {name}: {self.fK}")

    def ValidateX(self):
        if np.any(self.fx < 0.0) or np.any(self.fx > 1.0):
            debug_stop(f"Invalid molar fractions: {self.fx}")

    def ComputeK(self, Val):
        Val = np.asarray(Val, dtype=float)
        delta = Val[:2] - self.fRefVals

        if self.fIsInverse:
            inc = self.fK0 * np.dot(self.fDK0, delta)
            denominator = 1.0 + inc
            if abs(denominator) < 1e-30:
                debug_stop("Singular denominator in ComputeK")
            self.fK = self.fK0 / denominator
        else:
            inc = np.dot(self.fDK0, delta)
            self.fK = self.fK0 + inc

        self.ComputeDK(Val)

    def ComputeDK(self, Val):
        Val = np.asarray(Val, dtype=float)

        if self.fIsInverse:
            a = 1.0 + self.fK0 * np.dot(
                self.fDK0, Val[:2] - self.fRefVals
            )
            if abs(a) < 1e-30:
                debug_stop("Singular denominator in ComputeDK")
            self.fDK = -(self.fK0**2) * self.fDK0 / (a**2)
        else:
            self.fDK = self.fDK0.copy()

    def ValidateKDerivative(self, Val, dVal):
        Val = np.asarray(Val, dtype=float)
        dVal = np.asarray(dVal, dtype=float)

        self.ComputeK(Val)
        K = self.fK
        DK = self.fDK.copy()

        errors = []
        for i in range(10):
            alpha = (i + 1.0) / 10.0
            self.ComputeK(Val[:2] + alpha * dVal[:2])
            newK = self.fK

            guessK = np.dot(DK, alpha * dVal[:2])
            actualK = newK - K
            errors.append(abs(guessK - actualK))

        _print_errors("K", errors)

class Phase:
    def __init__(self):
        self.fS = 0.0
        self.fDS = np.zeros(5)
        self.fD = 0.0
        self.fDD = np.zeros(5)
        self.fP = 0.0
        self.frhoM = 0.0
        self.fDrhoM = np.zeros(2)

    def ValidateD(self):
        if not 0.0 <= self.fD <= 1.0:
            debug_stop(f"Invalid phase density fraction: {self.fD}")

    def ValidateS(self):
        if not 0.0 <= self.fS <= 1.0:
            debug_stop(f"Invalid saturation: {self.fS}")

class Compositional_H2O_CO2_NaCl:
    def __init__(self):
        self.fCO2 = Component()
        self.fH2O = Component()
        self.fNaCl = Component()

        self.fLiquidPhase = Phase()
        self.fGasPhase = Phase()
        self.fSolidPhase = Phase()

        self.frhoMT = 0.0
        self.fDrhoMT = np.zeros(5)

        self.fP = 0.0
        self.fT = 0.0

    def GetComponent(self, ComponentIdx):
        components = {
            ECO2: self.fCO2,
            EH2O: self.fH2O,
            ENaCl: self.fNaCl,
        }
        if ComponentIdx not in components:
            debug_stop("Invalid component index")
        return components[ComponentIdx]

    def SetZ(self, zCO2, zH2O, zNaCl):
        self.fCO2.fz = zCO2
        self.fH2O.fz = zH2O
        self.fNaCl.fz = zNaCl

    def SetRhoM(self, RhoLM, RhoGM, RhoSM):
        self.fLiquidPhase.frhoM = RhoLM
        self.fGasPhase.frhoM = RhoGM
        self.fSolidPhase.frhoM = RhoSM

    def GetX(self):
        return np.array([
            self.fCO2.fx[0], self.fCO2.fx[1],
            self.fH2O.fx[0], self.fH2O.fx[1],
            self.fNaCl.fx[0], self.fNaCl.fx[1],
        ])

    def SetX(self, xVals):
        xVals = np.asarray(xVals).reshape(6)
        self.fCO2.fx[:] = xVals[0:2]
        self.fH2O.fx[:] = xVals[2:4]
        self.fNaCl.fx[:] = xVals[4:6]

    def GetDX(self):
        DX = np.zeros((6, 2))
        DX[0:2, :] = self.fCO2.fDx
        DX[2:4, :] = self.fH2O.fDx
        DX[4:6, :] = self.fNaCl.fDx
        return DX

    def SetDX(self, i, DVec):
        DVec = np.asarray(DVec).reshape(6)
        self.fCO2.fDx[:, i] = DVec[0:2]
        self.fH2O.fDx[:, i] = DVec[2:4]
        self.fNaCl.fDx[:, i] = DVec[4:6]

    def GetD(self):
        return np.array([
            self.fLiquidPhase.fD,
            self.fGasPhase.fD,
            self.fSolidPhase.fD,
        ])

    def SetD(self, Val):
        Val = np.asarray(Val).reshape(3)
        self.fLiquidPhase.fD = Val[0]
        self.fGasPhase.fD = Val[1]
        self.fSolidPhase.fD = Val[2]

    def GetDD(self):
        return np.vstack([
            self.fLiquidPhase.fDD,
            self.fGasPhase.fDD,
            self.fSolidPhase.fDD,
        ])

    def SetDD(self, i, DVec):
        DVec = np.asarray(DVec).reshape(3)
        self.fLiquidPhase.fDD[i] = DVec[0]
        self.fGasPhase.fDD[i] = DVec[1]
        self.fSolidPhase.fDD[i] = DVec[2]

    def GetSRhoM(self):
        return np.array([
            self.fLiquidPhase.fS,
            self.fGasPhase.fS,
            self.fSolidPhase.fS,
            self.frhoMT,
        ])

    def GetDSRhoM(self):
        return np.vstack([
            self.fLiquidPhase.fDS,
            self.fGasPhase.fDS,
            self.fSolidPhase.fDS,
            self.fDrhoMT,
        ])

    def LoadState(self, P, T):
        self.fP = P
        self.fT = T
        PT = np.array([P, T], dtype=float)

        self.fCO2.ComputeK(PT)
        self.fH2O.ComputeK(PT)
        self.fNaCl.ComputeK(PT)

    def ComputeMolarFractions(self, maxIter=1000, tol=1e-10):
        # Unknowns: xCO2l, xCO2g, xH2Ol, xH2Og, xNaCll, xNaCls.
        A = np.zeros((6, 6))

        A[0, [0, 2, 4]] = 1.0
        A[1, [1, 3]] = 1.0
        A[2, 5] = 1.0

        A[3, 0] = 1.0
        A[3, 1] = -self.fCO2.fK

        A[4, 2] = 1.0
        A[4, 3] = -self.fH2O.fK

        A[5, 4] = 1.0
        A[5, 5] = -self.fNaCl.fK

        def residual():
            xCO2l, xCO2g, xH2Ol, xH2Og, xNaCll, xNaCls = self.GetX()

            return np.array([
                1.0 - xCO2l - xH2Ol - xNaCll,
                1.0 - xCO2g - xH2Og,
                1.0 - xNaCls,
                self.fCO2.fK * xCO2g - xCO2l,
                self.fH2O.fK * xH2Og - xH2Ol,
                self.fNaCl.fK * xNaCls - xNaCll,
            ])

        for _ in range(maxIter):
            R = residual()
            if norm(R) < tol:
                break

            try:
                correction = np.linalg.solve(A, R)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Singular matrix in ComputeMolarFractions"
                ) from exc

            self.SetX(self.GetX() + correction)
        else:
            debug_stop("ComputeMolarFractions did not converge")

        for component in (self.fCO2, self.fH2O, self.fNaCl):
            component.ValidateX()

        # Derivatives with respect to pressure and temperature.
        for i in range(2):
            DTanMat = np.zeros((6, 6))
            DTanMat[3, 1] = self.fCO2.fDK[i]
            DTanMat[4, 3] = self.fH2O.fDK[i]
            DTanMat[5, 5] = self.fNaCl.fDK[i]

            Result = DTanMat @ self.GetX()
            Result = np.linalg.solve(A, Result)
            self.SetDX(i, Result)

    def ComputeMolarDensityFractions(self):
        # Solve for Dl, Dg, Ds.
        xCO2l, xCO2g, xH2Ol, xH2Og, xNaCll, xNaCls = self.GetX()

        zCO2 = self.fCO2.fz
        zH2O = self.fH2O.fz
        zNaCl = self.fNaCl.fz

        A = np.eye(3)
        A[0, :] = [xCO2l, xCO2g, 0.0]
        A[1, :] = [xH2Ol, xH2Og, 0.0]
        A[2, :] = [xNaCll, 0.0, xNaCls]

        Z = np.array([zCO2, zH2O, zNaCl])

        try:
            D = np.linalg.solve(A, Z)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Singular matrix in ComputeMolarDensityFractions"
            ) from exc

        self.SetD(D)

        for phase in (
            self.fLiquidPhase, self.fGasPhase, self.fSolidPhase
        ):
            phase.ValidateD()

        # Derivatives with respect to P and T.
        for i in range(2):
            DTanMat = np.zeros((3, 3))
            DTanMat[0, 0] = self.fCO2.fDx[0, i]
            DTanMat[0, 1] = self.fCO2.fDx[1, i]
            DTanMat[1, 0] = self.fH2O.fDx[0, i]
            DTanMat[1, 1] = self.fH2O.fDx[1, i]
            DTanMat[2, 0] = self.fNaCl.fDx[0, i]
            DTanMat[2, 2] = self.fNaCl.fDx[1, i]

            Result = -np.linalg.solve(A, DTanMat @ D)
            self.SetDD(i, Result)

        # Derivatives with respect to zCO2, zH2O and zNaCl.
        inverseA = np.linalg.inv(A)
        for i in range(3):
            self.SetDD(i + 2, inverseA[:, i])

    def ComputeSaturationsAndTotalMolarDensity(self):
        rholM = self.fLiquidPhase.frhoM
        rhogM = self.fGasPhase.frhoM
        rhosM = self.fSolidPhase.frhoM

        Dl = self.fLiquidPhase.fD
        Dg = self.fGasPhase.fD
        Ds = self.fSolidPhase.fD

        A = np.array([
            [-rholM, -rhogM, -rhosM, 1.0],
            [-rholM, 0.0, 0.0, Dl],
            [0.0, 0.0, -rhosM, Ds],
            [1.0, 1.0, 1.0, 0.0],
        ])

        R = np.array([0.0, 0.0, 0.0, 1.0])

        try:
            result = np.linalg.solve(A, R)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "Singular matrix in saturation calculation"
            ) from exc

        self.fLiquidPhase.fS = result[0]
        self.fGasPhase.fS = result[1]
        self.fSolidPhase.fS = result[2]
        self.frhoMT = result[3]

        self.fLiquidPhase.ValidateS()
        self.fGasPhase.ValidateS()
        self.fSolidPhase.ValidateS()

        if self.frhoMT < 0.0:
            debug_stop("Negative total molar density")

        # Derivatives of saturations and total molar density.
        DD = self.GetDD()

        for i in range(5):
            dA = np.zeros((4, 4))
            dA[1, 3] = -DD[0, i]
            dA[2, 3] = -DD[2, i]

            temp = dA @ result
            temp = np.linalg.solve(A, temp)

            self.fLiquidPhase.fDS[i] = temp[0]
            self.fGasPhase.fDS[i] = temp[1]
            self.fSolidPhase.fDS[i] = temp[2]
            self.fDrhoMT[i] = temp[3]

    def ValidateXDerivative(self, Val, dVal):
        Val = np.asarray(Val, dtype=float)
        dVal = np.asarray(dVal, dtype=float)

        self.LoadState(Val[0], Val[1])
        self.ComputeMolarFractions()

        x = self.GetX()
        Dx = self.GetDX()
        errors = []

        for i in range(10):
            alpha = (i + 1.0) / 10.0
            self.LoadState(
                Val[0] + alpha * dVal[0],
                Val[1] + alpha * dVal[1],
            )
            self.ComputeMolarFractions()

            newX = self.GetX()
            guessX = Dx @ (alpha * dVal[:2])
            errors.append(norm(guessX - (newX - x)))

        _print_errors("x", errors)

    def ValidateDDerivative(self, Val, dVal):
        Val = np.asarray(Val, dtype=float)
        dVal = np.asarray(dVal, dtype=float)

        self.LoadState(Val[0], Val[1])
        self.SetZ(*Val[2:5])
        self.ComputeMolarFractions()
        self.ComputeMolarDensityFractions()

        D = self.GetD()
        DD = self.GetDD()
        errors = []

        for i in range(10):
            alpha = (i + 1.0) / 10.0
            newVal = Val + alpha * dVal

            self.LoadState(newVal[0], newVal[1])
            self.SetZ(*newVal[2:5])
            self.ComputeMolarFractions()
            self.ComputeMolarDensityFractions()

            newD = self.GetD()
            guessD = DD @ (alpha * dVal)
            errors.append(norm(guessD - (newD - D)))

        _print_errors("D", errors)

    def ValidateSDerivative(self, Val, dVal):
        Val = np.asarray(Val, dtype=float)
        dVal = np.asarray(dVal, dtype=float)

        self.LoadState(Val[0], Val[1])
        self.SetZ(*Val[2:5])
        self.ComputeMolarFractions()
        self.ComputeMolarDensityFractions()
        self.ComputeSaturationsAndTotalMolarDensity()

        SR = self.GetSRhoM()
        DSR = self.GetDSRhoM()
        errors = []

        for i in range(10):
            alpha = (i + 1.0) / 10.0
            newVal = Val + alpha * dVal

            self.LoadState(newVal[0], newVal[1])
            self.SetZ(*newVal[2:5])
            self.ComputeMolarFractions()
            self.ComputeMolarDensityFractions()
            self.ComputeSaturationsAndTotalMolarDensity()

            newSR = self.GetSRhoM()
            guessSR = DSR @ (alpha * dVal)
            errors.append(norm(guessSR - (newSR - SR)))

        _print_errors("S", errors)

    def PrintMolarFractions(self):
        self.fCO2.Print("CO2")
        self.fH2O.Print("H2O")
        self.fNaCl.Print("NaCl")

    def PrintMolarDensityFractions(self):
        print("Liquid phase D:", self.fLiquidPhase.fD)
        print("Gas phase D:", self.fGasPhase.fD)
        print("Solid phase D:", self.fSolidPhase.fD)

    def PrintSaturations(self):
        print("Liquid phase S:", self.fLiquidPhase.fS)
        print("Gas phase S:", self.fGasPhase.fS)
        print("Solid phase S:", self.fSolidPhase.fS)

    def PrintMolarDensity(self):
        print("Total molar density:", self.frhoMT)