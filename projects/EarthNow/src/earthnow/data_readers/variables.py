from pathlib import Path
from typing import Optional

import yaml
from dataclasses import dataclass, field

VARIABLES_YAML = Path(__file__).parent / "variables.yaml"


class VariableRegistry(dict):
    """Dict of alias -> ValidVariable, with lookups."""

    def register(self, variable: "ValidVariable") -> None:
        if variable.alias in self:
            raise ValueError(f"Error: The alias '{variable.alias}' is already registered")
        self[variable.alias] = variable

    @property
    def aliases(self) -> list[str]:
        return list(self.keys())

    @property
    def collections(self) -> list[str]:
        names = {name for var in self.values() for name in var.collection.collection}
        return sorted(names)

    def collection_aliases(self, collection_name: str) -> list[str]:
        """All aliases that define a variable name for the given collection."""
        return [
            alias
            for alias, var in self.items()
            if collection_name in var.collection.collection
        ]

    def resolve(self, alias: str, collection_name: str) -> str:
        """Returns "<varname>.<collection_name>" for alias, raising if either is invalid."""
        if alias not in self:
            raise KeyError(f"'{alias}' is not a registered alias")
        fullname = self[alias].collection.return_fullname(collection_name)
        if fullname is None:
            raise ValueError(
                f"'{alias}' has no variable registered for collection '{collection_name}'"
            )
        return fullname


VARIABLE_REGISTRY = VariableRegistry()


# Define a data class for the collections
@dataclass(frozen=True)
class CollectionVariable:
    # Maps collection name -> variable name within that collection.
    # Not all vars have all collections, and new collections require no code changes.
    collection: dict[str, str] = field(default_factory=dict)

    def return_fullname(self, collection_name: str) -> Optional[str]:
        """Returns the formatted string for a specific collection name if it exists."""
        value = self.collection.get(collection_name)
        if value:
            return f"{value}.{collection_name}"
        return None


# Define a data class for each variable with a nested collection class
@dataclass()
class ValidVariable:
    alias: str
    description: str
    collection: CollectionVariable = field(default_factory=CollectionVariable)


def return_fullname(input_dict: dict):
    return {
        key: VARIABLE_REGISTRY.resolve(key, value) for key, value in input_dict.items()
    }


# Load variable definitions and register them
with open(VARIABLES_YAML) as f:
    PRODUCTS_DICTIONARY = yaml.safe_load(f)

for alias, data in PRODUCTS_DICTIONARY.items():
    coll_var = CollectionVariable(collection=data["collection"])
    VARIABLE_REGISTRY.register(
        ValidVariable(alias=alias, description=data["description"], collection=coll_var)
    )
