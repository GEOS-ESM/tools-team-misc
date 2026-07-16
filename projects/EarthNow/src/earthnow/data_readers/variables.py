from pathlib import Path
from typing import Optional

import yaml
from dataclasses import dataclass, field

VARIABLES_YAML = Path(__file__).parent / "variables.yaml"


# Loader class to prevent duplicate overrides in yaml_load
class UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # loader.flatten_mapping(node)
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"Duplicate key '{key}' found in YAML file!")
            value = self.construct_object(value_node, deep=deep)
            mapping[key] = value

        return mapping


# Class to store each variable alias, description, and collection variable names
@dataclass()
class ValidVariable:
    alias: str
    description: str
    collections: dict[str, str] = field(default_factory=dict)

    def return_fullname(self, collection_name: str) -> Optional[str]:
        """Returns the formatted string for a specific collection name if it exists."""
        value = self.collections.get(collection_name)
        if value:
            return f"{value}.{collection_name}"
        return None


# Class to register variable aliases into registry
class VariableRegistry:
    """Registry of alias -> ValidVariable, with lookups."""

    def __init__(self):
        self._variables: dict[str, ValidVariable] = {}

    def __getitem__(self, alias: str) -> ValidVariable:
        return self._variables[alias]

    def __contains__(self, alias: str) -> bool:
        return alias in self._variables

    def register(self, variable: ValidVariable) -> None:
        if variable.alias in self._variables:
            raise ValueError(
                f"Error: The alias '{variable.alias}' is already registered"
            )
        self._variables[variable.alias] = variable

    @property
    def aliases(self) -> list[str]:
        return list(self._variables.keys())

    @property
    def alias_descriptions(self) -> dict[str, str]:
        """Returns a dictionary mapping alias -> description."""
        return {alias: var.description for alias, var in self._variables.items()}

    @property
    def collections(self) -> list[str]:
        """Returns a list of collections of the variable aliases"""
        names = {name for var in self._variables.values() for name in var.collections}
        return sorted(names)

    def collection_aliases(self, collection_name: str) -> list[str]:
        """All aliases that define a variable name for the given collection."""
        return [
            alias
            for alias, var in self._variables.items()
            if collection_name in var.collections
        ]

    def resolve(self, alias: str, collection_name: str) -> str:
        """Returns "<varname>.<collection_name>" for alias, raising if either is invalid."""
        if alias not in self._variables:
            raise KeyError(f"'{alias}' is not a registered alias")

        fullname = self._variables[alias].return_fullname(collection_name)
        if fullname is None:
            raise ValueError(
                f"'{alias}' has no variable registered for collection '{collection_name}'"
            )
        return fullname

    def resolve_many(self, alias_to_collection: dict[str, str]) -> dict[str, str]:
        """Batch form of resolve(): {alias: collection_name} -> {alias: fullname}."""
        return {
            alias: self.resolve(alias, collection_name)
            for alias, collection_name in alias_to_collection.items()
        }


def build_registry(yaml_path: Path) -> VariableRegistry:
    """Loads variable definitions from yaml_path and registers them."""
    with open(yaml_path) as f:
        products = yaml.load(f, Loader=UniqueKeyLoader)

    registry = VariableRegistry()
    for alias, data in products.items():
        registry.register(
            ValidVariable(
                alias=alias,
                description=data["description"],
                collections=data.get("collection", {}),
            )
        )
    return registry


VARIABLE_REGISTRY = build_registry(VARIABLES_YAML)
