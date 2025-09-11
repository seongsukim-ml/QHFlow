# Physical constants
# ================
# from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal, Angstrom, eV
# """Constants from ase.units"""
# Length unit (Angstrom in pyscf)
# ANG2BOHR = Angstrom / Bohr
# BOHR2ANG = 1.0 / ANG2BOHR

# Energy unit (Eh in pyscf)
# HA2eV = Hartree / eV
# eV2HA = eV / HA2eV
# HA2meV = 1000 * HA2eV
# meV2HA = 1000 * eV2HA

# KCALPM     = kcal / mol
# KCALPM2eV  = KCALPM / eV
# KCALPM2meV = KCALPM2eV * 1000

# HA2KCALPM  = Hartree / KCALPM     # Hartree to kcal/mol
# KCALPM2HA  = KCALPM / Hartree       # kcal/mol to Hartree

# ================
from ase.units import Bohr, Rydberg, kJ, kB, fs, Hartree, mol, kcal
# from ase.units import Angstrom, eV

# Default units
Angstrom = 1.0
eV = 1.0

# Length unit (Angstrom in pyscf)
ANG2BOHR = 1.8897261258369282     # Angstrom to Bohr conversion
BOHR2ANG = 0.5291772105638411     # Bohr to Angstrom conversion

# Energy unit (Eh in pyscf)
HA2eV    = 27.211396641308        # Hartree to eV conversion
HA2meV   = HA2eV * 1000           # Hartree to meV conversion
eV2HA    = 0.03674932247495664    # eV to Hartree
meV2HA   = eV2HA / 1000           # meV to Hartree

KCALPM2eV  = 0.04336410390059322   # kcal/mol to eV conversion
KCALPM2meV = KCALPM2eV * 1000      # kcal/mol to meV conversion
HA2KCALPM  = 627.5094738898777     # Hartree to kcal/mol
KCALPM2HA  = 0.001593601438080425  # kcal/mol to Hartree

# Force unit (Eh/Bohr in pyscf)
HA_BOHR_2_KCALPM_ANG = HA2KCALPM / BOHR2ANG        # Hartree/Bohr to kcal/mol/Angstrom
KCALPM_ANG_2_HA_BOHR = 1.0 / HA_BOHR_2_KCALPM_ANG  # kcal/mol/Angstrom to Hartree/Bohr
HA_BOHR_2_meV_ANG    = HA2meV / BOHR2ANG           # Hartree/Bohr to meV/Angstrom
meV_ANG_2_HA_BOHR    = 1.0 / HA_BOHR_2_meV_ANG     # meV/Angstrom to Hartree/Bohr

def print_unit_conversion():
    print(f"Length unit: {Angstrom} Angstrom")
    print(f"Bohr      : {1*BOHR2ANG} Angstrom")
    
    print(f"Energy unit: {eV} eV")
    print(f"Hartree   : {1*HA2eV} eV")
    print(f"meV       : {1000*eV} meV")
    print(f"kcal/mol  : {1*KCALPM2meV} meV")
    print(f"Hartree   : {1*HA2meV} meV")
