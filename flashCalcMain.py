import numpy as np

from flashCalcClasses import ECO2, EH2O, ENaCl, Compositional_H2O_CO2_NaCl

def main():
    compositional = Compositional_H2O_CO2_NaCl()

    GradKCO2 = np.array([1.0, 0.0])
    GradKH2O = np.array([-1.0, 0.05])
    GradKNaCl = np.array([0.000002, 0.0005])

    CO2 = compositional.GetComponent(ECO2)
    H2O = compositional.GetComponent(EH2O)
    NaCl = compositional.GetComponent(ENaCl)

    CO2.SetK(1.0e-4, GradKCO2, True)
    H2O.SetK(1.0e4, GradKH2O, False)
    NaCl.SetK(0.0, GradKNaCl, False)

    RefVals = np.array([1.0e6, 300.0])

    CO2.SetReferenceValues(RefVals)
    H2O.SetReferenceValues(RefVals)
    NaCl.SetReferenceValues(RefVals)

    compositional.SetZ(0.4, 0.5, 0.1)
    compositional.SetRhoM(1600.0, 1000.0, 2000.0)

    Val = np.array([1e6, 300.0, 0.4, 0.4, 0.2])
    DeltaVal = np.array([1000.0, 10.0, 0.04, -0.03, -0.01])

    # Validate K derivatives.
    CO2.ValidateKDerivative(Val, DeltaVal)
    H2O.ValidateKDerivative(Val, DeltaVal)
    NaCl.ValidateKDerivative(Val, DeltaVal)

    # Validate molar fraction derivatives.
    compositional.ValidateXDerivative(Val, DeltaVal)

    # Validate molar density fraction derivatives.
    compositional.ValidateDDerivative(Val, DeltaVal)

    # Validate saturation and total molar density derivatives.
    compositional.ValidateSDerivative(Val, DeltaVal)

    # Compute and print the state.
    compositional.LoadState(Val[0], Val[1])
    compositional.SetZ(*Val[2:5])

    compositional.ComputeMolarFractions()
    compositional.PrintMolarFractions()

    compositional.ComputeMolarDensityFractions()
    compositional.PrintMolarDensityFractions()

    compositional.ComputeSaturationsAndTotalMolarDensity()
    compositional.PrintSaturations()
    compositional.PrintMolarDensity()

    print("DX:\n", compositional.GetDX())
    print("DD:\n", compositional.GetDD())
    print("DS and total molar density derivatives:\n", compositional.GetDSRhoM())

if __name__ == "__main__":
    main()