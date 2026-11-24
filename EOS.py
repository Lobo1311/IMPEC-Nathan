import numpy as np
from abc import abstractmethod, ABC

R = 8.31446261815324 # Universal gas constant in J/(mol*K)

class CubicEOS:
    def __init__(self, params):
        self.fParams:EOSParams = params
        self.fn = 2 #* number of components (CO2 and H2O)

        self.faPhase = np.zeros(self.fn) #* EOS parameter a for each phase
        self.fbPhase = np.zeros(self.fn) #* EOS parameter b for each phase
        self.faComp = np.zeros(self.fn) #* EOS parameter a for each component
        self.fbComp = np.zeros(self.fn) #* EOS parameter b for each component\
        self.faProduct = np.zeros((self.fn, self.fn)) #* EOS parameter a for each component pair
        self.fA = np.zeros(self.fn) #* dimensionless EOS parameter A for each phase
        self.fB = np.zeros(self.fn) #* dimensionless EOS parameter B for each phase
        self.fZ = np.zeros(self.fn) #* compressibility factors for each phase

        self.fUseBinaryParameters = False #* flag to indicate whether to use binary interaction parameters

    def CalculateAlpha(self, T):
        self.fParams.CalculateLambda()
        self.fAlpha = np.zeros(len(self.fParams.fomega))

        for i in range(len(self.fParams.fomega)):
            alphaValue = self.fParams.CalculateAlpha(T, i)
            self.fAlpha[i] = alphaValue

    def CalculateA(self, T, i):
        ac = self.fParams.fOmegaA * R ** 2 * self.fParams.fTc[i] ** self.fParams.fTemperatureExp / self.fParams.fPc[i]

        if self.fParams.fIsPolar[i]: #* Mathias-Copeman expression
            Tr = T / self.fParams.fTc[i]
            c1, c2, c3 = self.fParams.fc
            if Tr <= 1.0:
                return ac * (1 + c1 * (1 - Tr**0.5) + c2 * (1 - Tr**0.5)**2 + c3*(1 - Tr**0.5)**3) ** 2
            else:
                return ac * (1 + c1 * (1 - Tr**0.5)) ** 2
        else: #* Soave-type expression
            return ac * self.fAlpha[i]

    def CalculateKij(self, T): #* binary interaction parameter
        if not self.fUseBinaryParameters:
            return np.zeros((self.fn, self.fn))

        if T < 278 or T > 623:
            raise ValueError("Temperature is out of the valid range for kij calculation (278 K to 623 K).")

        if T < 373:
            A = -0.01363
            B = 0.00088
        else:
            A = 0.39283
            B = -0.000313
        
        val = A + B * T

        return np.array([[0, val], [val, 0]])

    def CalculateLij(self, T):
        if not self.fUseBinaryParameters:
            return np.zeros((self.fn, self.fn))

        if T < 278 or T > 623:
            raise ValueError("Temperature out of valid range.")

        if T < 373:
            A = 0.27752
            B = -0.000197
        else:
            A = 0.33147
            B = -0.000436

        val = A + B * T

        return np.array([
            [0.0, val],
            [val, 0.0]
        ])

    def CalculateCompressibilityFactors(self, P, T, x):
        self.CalculateAlpha(T)
        kij = self.CalculateKij(T)
        lij = self.CalculateLij(T)

        for iComp in range(self.fn): #* 0 for CO2, 1 for H2O
            self.faComp[iComp] = self.CalculateA(T, iComp)
            self.fbComp[iComp] = self.fParams.fOmegaB * R * self.fParams.fTc[iComp] / self.fParams.fPc[iComp]

        self.faProduct = np.sqrt(np.outer(self.faComp, self.faComp))
        bProduct = 0.5 * (np.add.outer(self.fbComp, self.fbComp))

        for iPhase in range(self.fn): #* 0 for liquid, 1 for gas
            xPhase = np.array([x[iPhase], x[iPhase + 2]])
            xProduct = np.outer(xPhase, xPhase)

            aPhase = np.sum(np.multiply(np.multiply(self.faProduct, xProduct), (1 - kij)))
            bPhase = np.sum(xProduct * bProduct * (1 - lij))

            self.faPhase[iPhase] = aPhase
            self.fbPhase[iPhase] = bPhase

            APhase = aPhase * P / (R ** 2 * T ** self.fParams.fTemperatureExp)
            BPhase = bPhase * P / (R * T)

            self.fA[iPhase] = APhase
            self.fB[iPhase] = BPhase

            #* Calculate the fugacity coefficient using an EOS
            Z = self.SolveCubicEOS(APhase, BPhase, iPhase)
            self.fZ[iPhase] = Z    

    def CalculateKRatios(self, P, T, x):
        self.CalculateCompressibilityFactors(P, T, x)
        
        KRatios = [] #* xl / xg for each component
        phiMat = np.zeros((self.fn, self.fn)) #* fugacity coefficients for each component in each phase
        xMat = np.array([[x[0], x[2]], [x[1], x[3]]]) #* mole fractions for each component in each phase
        kij = self.CalculateKij(T)

        for iPhase in range(self.fn):
            for iComp in range(self.fn):
                lnphi = self.fParams.CalculateLnPhi(iComp, iPhase, kij, self.fA, self.fB, self.faPhase, self.fbPhase, self.fbComp, self.faProduct, xMat, self.fZ)
                phi = np.exp(lnphi)
                phiMat[iPhase, iComp] = phi
            
        for i in range(self.fn):
            KRatios.append(phiMat[0, i] / phiMat[1, i])

        return KRatios

    def CalculatePhaseMolarDensity(self, P, T, x):
        self.CalculateCompressibilityFactors(P, T, x)
        rhoMolar = P / (self.fZ * R * T)

        return rhoMolar

    def SolveCubicEOS(self, A, B, i):
        #* Coefficients of the cubic equation: a*Z^3 + b*Z^2 + c*Z + d = 0
        coeffs = self.fParams.fCoeff(A, B)
        roots = np.roots(coeffs)

        #* Return the real root (the one that corresponds to the physical solution)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        possible_roots = real_roots[real_roots > B]

        if len(possible_roots) == 0:
            raise ValueError("No valid real roots found for the cubic equation of state.") 

        if (i == 0):  #* For liquid phase, return the smallest possible real root
            return min(possible_roots)
        
        elif (i == 1):  #* For gas phase, return the largest possible real root
            return max(possible_roots)

        else:
            raise ValueError("Invalid phase index. Use 0 for liquid and 1 for gas.")


class EOSParams(ABC):
    def __init__(self, omegaCO2 = 0.228, omegaH2O = 0.344, TcCO2 = 304.19, TcH2O = 647.14, PcCO2 = 7.38e6, PcH2O = 22.064e6):
        self.fOmegaA = 0.0 #* EOS parameter for the component
        self.fOmegaB = 0.0 #* EOS parameter for the component
        self.fomega = [omegaCO2, omegaH2O] #* acentric factor for each component (CO2, H2O in this case)
        self.fTc = [TcCO2, TcH2O] #* critical temperature for each component (CO2, H2O in this case)
        self.fPc = [PcCO2, PcH2O] #* critical pressure for each component (CO2, H2O in this case)
        self.fIsPolar = [False, True] #* indicates if the component is polar (CO2 is non-polar, H2O is polar)
        self.fc = [0.9236, -0.3794, 0.4424]
        self.fTemperatureExp = 2.0
        self.fCoeff:callable

    def SetCubicCoeff(self, CoeffFunc):
        self.fCoeff = CoeffFunc

    def CalculateLambda(self):
        self.fLambda = np.zeros(len(self.fomega))

        for i, omega in enumerate(self.fomega):
            lambdaValue = 0.37464 + 1.5423 * omega - 0.26992 * omega ** 2
            self.fLambda[i] = lambdaValue

    @abstractmethod
    def CalculateAlpha(self, T, i):
        raise ValueError("Must be implemented on derived class")

    @abstractmethod
    def CalculateLnPhi(self, iComp, iPhase, kij, AVec, BVec, aVec, bVec, bComponent, aProduct, xMat, ZVec):
        raise ValueError("Must be implemented on derived class")

class PengRobinsonParams(EOSParams):
    def __init__(self, omegaCO2=0.228, omegaH2O=0.344, TcCO2=304.19, TcH2O=647.14, PcCO2=7380000, PcH2O=22064000):
        super().__init__(omegaCO2, omegaH2O, TcCO2, TcH2O, PcCO2, PcH2O)

        self.fOmegaA = 0.45724
        self.fOmegaB = 0.077796

        def PRCoeff(A, B):
            #* This function solves the cubic equation of state for Z (compressibility factor)
            #* Using the Peng-Robinson EOS
            a = 1
            b = -(1 - B)
            c = A - 2 * B - 3 * B ** 2
            d = -(A * B - B ** 2 - B ** 3)

            return [a, b, c, d]

        self.fCoeff = PRCoeff

    def CalculateAlpha(self, T, i):
        return (1 + self.fLambda[i] * (1 - (T / self.fTc[i]) ** 0.5)) ** 2

    def CalculateLnPhi(self, iComp, iPhase, kij, AVec, BVec, aVec, bVec, bComponent, aProduct, xMat, ZVec):
        t1 = bComponent[iComp] / bVec[iPhase] * (ZVec[iPhase] - 1)
        t2 = - np.log(ZVec[iPhase] - BVec[iPhase])
        t31 = - AVec[iPhase] / (2 * np.sqrt(2) * BVec[iPhase]) * (2 / aVec[iPhase] * np.sum(aProduct[iComp] * xMat[iPhase] * (1 - kij[iComp])) - bComponent[iComp] / bVec[iPhase])
        t32 = np.log((ZVec[iPhase] + (1 + np.sqrt(2)) * BVec[iPhase]) / (ZVec[iPhase] + (1 - np.sqrt(2)) * BVec[iPhase]))

        return t1 + t2 + t31 * t32


class ModRedlichKwongParams(EOSParams):
    def __init__(self, omegaCO2=0.228, omegaH2O=0.344, TcCO2=304.19, TcH2O=647.14, PcCO2=7380000, PcH2O=22064000):
        super().__init__(omegaCO2, omegaH2O, TcCO2, TcH2O, PcCO2, PcH2O)

        self.fOmegaA = 0.42748
        self.fOmegaB = 0.08664

        def RKCoeff(A, B):
            #* This function solves the cubic equation of state for Z (compressibility factor)
            #* Using the Redlich-Kwong EOS
            a = 1
            b = -1
            c = A - B - B ** 2
            d = -A * B

            return [a, b, c, d]

        self.fCoeff = RKCoeff

    def CalculateAlpha(self, T, i):
        return T / self.fTc[i]

    def CalculateLnPhi(self, iComp, iPhase, kij, AVec, BVec, aVec, bVec, bComponent, aProduct, xMat, ZVec):
        t1 = bComponent[iComp] / bVec[iPhase] * (ZVec[iPhase] - 1)
        t2 = - np.log(ZVec[iPhase] - BVec[iPhase])
        t31 = - AVec[iPhase] / BVec[iPhase] * (2 / aVec[iPhase] * np.sum(aProduct[iComp] * xMat[iPhase] * (1 - kij[iComp])) - bComponent[iComp] / bVec[iPhase])
        t32 = np.log((ZVec[iPhase] + BVec[iPhase]) / ZVec[iPhase])

        return t1 + t2 + t31 * t32

class RedlichKwongParamsV2(ModRedlichKwongParams):
    def __init__(self, omegaCO2=0.228, omegaH2O=0.344, TcCO2=304.19, TcH2O=647.14, PcCO2=7380000, PcH2O=22064000):
        super().__init__(omegaCO2, omegaH2O, TcCO2, TcH2O, PcCO2, PcH2O)

        self.fTemperatureExp = 2.5

    def CalculateAlpha(self, T, i):
        return (self.fTc[i] / T)**0.5

class RedlichKwongSoaveParams(ModRedlichKwongParams):
    def __init__(self, omegaCO2=0.228, omegaH2O=0.344, TcCO2=304.19, TcH2O=647.14, PcCO2=7380000, PcH2O=22064000):
        super().__init__(omegaCO2, omegaH2O, TcCO2, TcH2O, PcCO2, PcH2O)

    def CalculateAlpha(self, T, i):
        return (1 + self.fLambda[i] * (1 - (T / self.fTc[i]) ** 0.5)) ** 2

    def CalculateLambda(self):
        self.fLambda = np.zeros(len(self.fomega))

        for i, omega in enumerate(self.fomega):
            lambdaValue = 0.48 + 1.574 * omega - 0.176 * omega ** 2
            self.fLambda[i] = lambdaValue

if __name__ == "__main__":
    # Example usage
    params = PengRobinsonParams()
    eos = CubicEOS(params)
    P = 10.0e6  # Pressure in Pa
    T = 313.15  # Temperature in K
    x = np.array([0.3, 0.6, 0.7, 0.4])  # Mole fractions (co2 l, co2 g, h2o l, h2o g)
    x = np.array([0.021722781329965243, 0.9956958962440414, 0.9782772186700348, 0.004304103755958577])

    eos.CalculateCompressibilityFactors(P, T, x)
    print("Compressibility factors:", eos.fZ)
    print("K ratios:", eos.CalculateKRatios(P, T, x))
    print("Phase molar densities:", eos.CalculatePhaseMolarDensity(P, T, x))
