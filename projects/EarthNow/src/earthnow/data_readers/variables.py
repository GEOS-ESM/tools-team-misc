from dataclasses import dataclass

VARIABLE_REGISTRY = {}


@dataclass()
class ValidVariables:
    alias: str
    description: str

    def __post_init__(self):
        # Check for duplicates
        if self.alias in VARIABLE_REGISTRY:
            raise ValueError(f"Error: The alias '{self.alias}' is already registered")

        # If it's not a duplicate, add it to the registry
        VARIABLE_REGISTRY[self.alias] = self


# Register variable aliases and description (NO DUPLICATES)
ValidVariables(alias="VORT500", description="500 mb Relative Vorticity (s-1)")
ValidVariables(alias="H500", description="Height at 500 hPa (m)")
ValidVariables(alias="T2M", description="2-Meter Temperature (K)")
ValidVariables(
    alias="DBZ_MAX", description="Maximum Composite Radar Reflectivity (dBZ)"
)
ValidVariables(alias="UH25", description="Updraft Helicity 2-5 KM (m2 s-2)")
ValidVariables(alias="CAPE", description="Convective Available Potential Energy (J/kg)")


# Tests
# print(VARIABLE_REGISTRY["VORT500"].alias)
# print(VARIABLE_REGISTRY)
# test = {
#     VARIABLE_REGISTRY["VORT500"].alias: "VORT500.inst1_2d_asm_Nx",
#     VARIABLE_REGISTRY["T2M"].alias: "T2M.inst1_2d_asm_Nx",
# }
# print(test)
# print(test["VORT500"])

# Maybe it just easier to make a dictionary and import it?
# variables_test = {
#     "VORT500": {"alias": "VORT500", "description": "500 mb Relative Vorticity"},
#     "T2M": {"alias": "T2M", "description": "2-meter Temperature"},
#     # {"alias": INSERT_ALIAS_HERE, "description": INSERT_DESCRIPTION HERE},
# }
# print(variables_test["VORT500"]["alias"]) # This is ok but maybe the class is safer? Checks for dupes? We can add more features later

# Function definition? This would force every variable registered to have an alias : variable.collection
# def assign_aliases():
#     """Assign valid aliases to true variable.collection"""
