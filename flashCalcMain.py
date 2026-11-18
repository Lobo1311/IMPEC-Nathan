import numpy as np

from flashCalcClasses import ECO2, EH2O, ENaCl, Compositional_H2O_CO2_NaCl
from EOS import *

def Validate():
    compositional = Compositional_H2O_CO2_NaCl()

    GradKCO2 = np.array([1.0, 0.0])
    GradKH2O = np.array([-1.0, 0.05])
    GradKNaCl = np.array([0.000002, 0.0005])

    CO2 = compositional.GetComponent(ECO2)
    H2O = compositional.GetComponent(EH2O)
    NaCl = compositional.GetComponent(ENaCl)

    CO2.SetK(1.0e4, GradKCO2, False)
    H2O.SetK(1.0e-4, GradKH2O, True)
    NaCl.SetK(1e30, GradKNaCl, False)

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

def Run():
    compositional = Compositional_H2O_CO2_NaCl()

    Val = np.array([1e6, 300.0, 0.4, 0.4, 0.2])
    DeltaVal = np.array([1000.0, 10.0, 0.04, -0.03, -0.01])

    #* CO2 L, CO2 G, H2O L, H2O G, NaCl L, NaCl S
    compositional.SetX([0.05, 0.90, 0.95, 0.10, 0.0, 1.0])

    RefVals = np.array([1.0e6, 300.0])
    compositional.LoadK(Val[0], Val[1])

    CO2 = compositional.GetComponent(ECO2)
    H2O = compositional.GetComponent(EH2O)
    NaCl = compositional.GetComponent(ENaCl)

    CO2.SetReferenceValues(RefVals)
    H2O.SetReferenceValues(RefVals)
    NaCl.SetReferenceValues(RefVals)

    compositional.SetZ(0.4, 0.5, 0.1)
    compositional.SetRhoM(1600.0, 1000.0, 2000.0)

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

def ComparingSpycherPruess():
    from impecmrst import flash, spycher_pruess_eq, Params
    params = Params()
    #z_co2 = 1e-12
    z_co2 = 0.4
    z_h2o = 1.0 - z_co2
    z_nacl = 0.0
    V_impec, Sl_impec, Sg_impec, x_l_co2_impec, x_g_co2_impec, rho_l_molar_impec, rho_g_molar_impec = flash(z_co2, params.p_res, params.T_K, params)

    x_co2_l, x_co2_g = spycher_pruess_eq(params.p_res, params.T_K, params)
    x_h2o_l = 1.0 - x_co2_l
    x_h2o_g = 1.0 - x_co2_g
    x_nacl_l = 0.0
    x_nacl_s = 1.0

    compositional = Compositional_H2O_CO2_NaCl()
    compositional.fP = params.p_res
    compositional.fT = params.T_K
    compositional.SetX([x_co2_l, x_co2_g, x_h2o_l, x_h2o_g, x_nacl_l, x_nacl_s])
    compositional.SetZ(z_co2, z_h2o, z_nacl)
    compositional.SetRhoM(1600.0, 1000.0, 2000.0)

    compositional.ComputeMolarDensityFractions()
    compositional.ComputeSaturationsAndTotalMolarDensity()

    import numpy.testing as npt
    npt.assert_almost_equal(V_impec, compositional.fGasPhase.fD, 5, "Gas phase molar density fraction mismatch")
    npt.assert_almost_equal(Sl_impec, compositional.fLiquidPhase.fS, 5, "Liquid saturation mismatch")
    npt.assert_almost_equal(Sg_impec, compositional.fGasPhase.fS, 5, "Gas saturation mismatch")
    npt.assert_almost_equal(x_l_co2_impec, compositional.fCO2.fx[0], 5, "Liquid CO2 mole fraction mismatch")
    npt.assert_almost_equal(x_g_co2_impec, compositional.fCO2.fx[1], 5, "Gas CO2 mole fraction mismatch")
    #npt.assert_almost_equal(rho_l_molar_impec, compositional.fLiquidPhase.frhoM, 5, "Liquid molar density mismatch")
    #npt.assert_almost_equal(rho_g_molar_impec, compositional.fGasPhase.frhoM, 5, "Gas molar density mismatch")

if __name__ == "__main__":
    ComparingSpycherPruess()