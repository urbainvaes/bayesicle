""" Convenience module to provide access to all solvers with only one import.
"""

import solver_eks
import solver_md
import solver_cbo_cbs

CboSolver = solver_cbo_cbs.CboSolver
CbsSolver = solver_cbo_cbs.CbsSolver
EksSolver = solver_eks.EksSolver

MdSolver = solver_md.MdSolver
MdSimulation = solver_md.MdSimulation
