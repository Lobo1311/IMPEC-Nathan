import numpy as np
R = 8.31446261815324 # Universal gas constant in J/(mol*K)

class CubicEOS:
    def __init__(self, params):
        self.params:EOSParams = params

    def CalculateAlpha(self, T):
        self.params.CalculateLambda()
        self.fAlpha = []

        for i in range(len(self.params.fomega)):
            alphaValue = (1 + self.params.fLambda[i] * (1 - (T / self.params.fTc[i]) ** 0.5)) ** 2
            self.fAlpha.append(alphaValue)

    def CalculateA(self, T, i):
        ac = self.params.fOmegaA * (R * self.params.fTc[i]) ** 2 / self.params.fPc[i]

        if self.params.fIsPolar: #* Mathias-Copeman expression
            Tr = T / self.params.fTc[i]
            c1, c2, c3 = self.params.fc
            if Tr <= 1.0:
                return ac * (1 + c1 * (1 - Tr**0.5) + c2 * (1 - Tr**0.5)**2 + c3*(1 - Tr**0.5)**3) ** 2
            else:
                return ac * (1 + c1 * (1 - Tr**0.5)) ** 2
        else: #* Soave-type expression
            return ac * self.fAlpha[i]

    def CalculateKij(self, T): #* binary interaction parameter
        return np.zeros(2, 2) 
        if T < 278 or T > 623:
            raise ValueError("Temperature is out of the valid range for kij calculation (278 K to 623 K).")

        if T < 373:
            A = -0.01363
            B = 0.00088
        else:
            A = 0.39283
            B = -0.000313
        
        return A + B * T

    def PlotKij(self):
        T_values = np.linspace(278, 623, 100)
        kij_values = [self.CalculateKij(T) for T in T_values]

        import matplotlib.pyplot as plt
        plt.plot(T_values, kij_values)
        plt.xlabel('Temperature (K)')
        plt.ylabel('Binary Interaction Parameter (kij)')
        plt.title('Binary Interaction Parameter vs Temperature')
        plt.grid()
        plt.show()

    def CalculateKRatios(self, T, P, x):
        self.CalculateAlpha(T)
        KRatios = [] #* xl / xg for each component
        phiMat = np.zeros((2, 2)) #* fugacity coefficients for each component in each phase

        for iPhase in range(2): #* 0 for liquid, 1 for gas
            aComponent = []
            bComponent = []
            for iComp in range(2): #* 0 for CO2, 1 for H2O
                aComponent.append(self.CalculateA(T, iComp))
                bComponent.append(self.params.fOmegaB * R * self.params.fTc[iComp] / self.params.fPc[iComp])

            aProduct = np.array([[aComponent[0], np.sqrt(aComponent[0] * aComponent[1])],
                                    [np.sqrt(aComponent[1] * aComponent[0]), aComponent[1]]])
            xProduct = np.array([[x[iPhase]**2, x[iPhase] * x[2 + iPhase]],
                                  [x[2 + iPhase] * x[iPhase], x[2 + iPhase]**2]])
            
            aPhase = np.sum(np.multiply(np.multiply(aProduct, xProduct), (1 - self.CalculateKij(T))))
            bPhase = x[iPhase] * bComponent[0] + x[2 + iPhase] * bComponent[1]

            APhase = aPhase * P / (R * T) ** 2
            BPhase = bPhase * P / (R * T)

            #* Calculate the fugacity coefficient using the Peng-Robinson EOS
            Z = self.SolveCubicEOS(APhase, BPhase)

            for iComp in range(2):
                lnphi = bComponent[iComp] / bPhase * (Z - 1) - np.log(Z - BPhase) - APhase / (2 * np.sqrt(2) * BPhase) * (2 / aPhase * np.sum(np.multiply(aProduct[iComp, :], xProduct[iComp, :]) * (1 - self.CalculateKij(T))) - bComponent[iComp] / bPhase) * np.log((Z + (1 + np.sqrt(2)) * BPhase) / (Z - (1 - np.sqrt(2)) * BPhase))

                phi = np.exp(lnphi)
                phiMat[iPhase, iComp] = phi
            
        for i in range(2):
            KRatios.append(phiMat[1, i] / phiMat[0, i])

        return KRatios

    def SolveCubicEOS(self, A, B):
        #* This function solves the cubic equation of state for Z (compressibility factor)
        #* Using the Peng-Robinson EOS as an example
        a = 1
        b = -(1 - B)
        c = A - 2 * B - 3 * B ** 2
        d = -(A * B - B ** 2 - B ** 3)

        #* Coefficients of the cubic equation: Z^3 + b*Z^2 + c*Z + d = 0
        coeffs = [a, b, c, d]
        roots = np.roots(coeffs)

        #* Return the real root (the one that corresponds to the physical solution)
        real_roots = roots[np.isreal(roots)].real
        return max(real_roots)  # Return the largest real root as Z


class EOSParams:
    def __init__(self, omegaCO2 = 0.228, omegaH2O = 0.344, TcCO2 = 304.19, TcH2O = 647.14, PcCO2 = 7.38e6, PcH2O = 22.064e6):
        self.fOmegaA = 0.45724 #* EOS parameter for the component
        self.fOmegaB = 0.077796 #* EOS parameter for the component
        self.fomega = [omegaCO2, omegaH2O] #* acentric factor for each component (CO2, H2O in this case)
        self.fTc = [TcCO2, TcH2O] #* critical temperature for each component (CO2, H2O in this case)
        self.fPc = [PcCO2, PcH2O] #* critical pressure for each component (CO2, H2O in this case)
        self.fIsPolar = [False, True] #* indicates if the component is polar (CO2 is non-polar, H2O is polar)
        self.fc = [0.9236, -0.3794, 0.4424]
        

    def CalculateLambda(self):
        self.fLambda = []

        for omega in self.fomega:
            lambdaValue = 0.37464 + 1.5423 * omega - 0.26992 * omega ** 2
            self.fLambda.append(lambdaValue)

cubic_eos = CubicEOS(EOSParams())
cubic_eos.PlotKij()  # This will plot the binary interaction parameter vs temperature