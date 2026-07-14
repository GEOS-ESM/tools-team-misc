from typing import Optional
from dataclasses import dataclass, field, fields

"""
TEMPLATE TO ADD PRODUCTS TO THE DICTIONARY BELOW, THEY WILL BE AUTOMATICALLY ADDED TO VARIABLE_REGISTRY
    "ALIAS": {
        "alias": "ALIAS", # The alias should match that used in inst1_2d_asm_Nx
        "description": "DESCRIPTION",  # Long name description
        "collection": {
            "inst1_2d_asm_Nx": "VARIABLE_NAME",
            "hwt_15mn_slv_LCC": "VARIABLE_NAME",
            "hwt_30mn_slv_LCC": "VARIABLE_NAME",
        },
"""
PRODUCTS_DICTIONARY = {
    "CAPE": {
        "alias": "CAPE",
        "description": "Convective Available Potential Energy (J/kg)",
        "collection": {
            "inst1_2d_asm_Nx": "CAPE",
            "hwt_15mn_slv_LCC": "CAPE",
            "hwt_30mn_slv_LCC": "CAPE",
        },
    },
    "DBZ_MAX": {
        "alias": "DBZ_MAX",
        "description": "Maximum Composite Radar Reflectivity",
        "collection": {
            "inst1_2d_asm_Nx": "DBZ_MAX",
            "hwt_15mn_slv_LCC": "REFC",
            "hwt_30mn_slv_LCC": "REFC",
        },
    },
    "H500": {
        "alias": "H500",
        "description": "Height at 500 hPa (m)",
        "collection": {
            "inst1_2d_asm_Nx": "H500",
            "hwt_15mn_slv_LCC": "H500",
            "hwt_30mn_slv_LCC": "H500",
        },
    },
    "HGT_SFC": {
        "alias": "HGT_SFC",
        "description": "",
        "collection": {
            "hwt_15mn_slv_LCC": "HGT_SFC",
            "hwt_30mn_slv_LCC": "HGT_SFC",
        },
    },
    "ICE": {
        "alias": "ICE",
        "description": "",
        "collection": {
            "inst1_2d_asm_Nx": "ICE",
            "hwt_15mn_slv_LCC": "ICE",
            "hwt_30mn_slv_LCC": "ICE",
        },
    },
    "RAIN": {
        "alias": "RAIN",
        "description": "",
        "collection": {
            "inst1_2d_asm_Nx": "RAIN",
            "hwt_15mn_slv_LCC": "RAIN",
            "hwt_30mn_slv_LCC": "RAIN",
        },
    },
    "SNOW": {
        "alias": "SNOW",
        "description": "",
        "collection": {
            "inst1_2d_asm_Nx": "SNOW",
            "hwt_15mn_slv_LCC": "SNOW",
            "hwt_30mn_slv_LCC": "SNOW",
        },
    },
    "T2M": {
        "alias": "T2M",
        "description": "2 Meter Temperature",
        "collection": {
            "inst1_2d_asm_Nx": "TMP",
            "hwt_15mn_slv_LCC": "TMP_2M",
            "hwt_30mn_slv_LCC": "TMP_2M",
        },
    },
    "UH25": {
        "alias": "UH25",
        "description": "Updraft Helicity 2-5 KM (m2 s-2)",
        "collection": {
            "inst1_2d_asm_Nx": "UH25",
            "hwt_15mn_slv_LCC": "UPHL_2-KM",
            "hwt_30mn_slv_LCC": "UPHL_2-KM",
        },
    },
    "VORT500": {
        "alias": "VORT500",
        "description": "500 mb Relative Vorticity",
        "collection": {
            "inst1_2d_asm_Nx": "VORT500",
            "hwt_15mn_slv_LCC": "VORT500",
            "hwt_30mn_slv_LCC": "VORT500",
        },
    },
}

VARIABLE_REGISTRY = {}


# Define a data class for the collections
@dataclass(frozen=True)
class CollectionVariable:
    # Each colleciton is optional since not all vars have all collections
    inst1_2d_asm_Nx: Optional[str] = None
    hwt_15mn_slv_LCC: Optional[str] = None
    hwt_30mn_slv_LCC: Optional[str] = None

    def return_fullname(self, collection_name: str) -> Optional[str]:
        """Returns the formatted string for a specific collection name if it exists."""
        value = getattr(self, collection_name, None)
        if value:
            return f"{value}.{collection_name}"
        return None


# Define a data class for each variable with a nested collection class
@dataclass()
class ValidVariable:
    alias: str
    description: str
    collection: list[CollectionVariable] = field(default_factory=CollectionVariable)

    def __post_init__(self):
        # Check for duplicates, every alias must be unique
        if self.alias in VARIABLE_REGISTRY:
            raise ValueError(f"Error: The alias '{self.alias}' is already registered")
        VARIABLE_REGISTRY[self.alias] = self


def return_fullname(input_dict: dict):
    return {
        key: VARIABLE_REGISTRY[key].collection.return_fullname(value)
        for key, value in input_dict.items()
    }


# Add dictionaries to the class registry
for key, data in PRODUCTS_DICTIONARY.items():
    coll_var = CollectionVariable(**data["collection"])
    ValidVariable(
        alias=data["alias"], description=data["description"], collection=coll_var
    )
