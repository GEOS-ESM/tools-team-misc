# aliases.py
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List, Any, Optional
import argparse
import logging

logger = logging.getLogger('cfapi.aliases')

def _norm(s: str) -> str:
    # casefold handles unicode edge cases better than lower()
    return s.casefold()

@dataclass
class Aliases:
    """_summary_
    input - [(Truename1, alias1,tn1), (Truename2, alias,tn2)]
    Raises:
        argparse.ArgumentTypeError: _description_
        argparse.ArgumentTypeError: _description_

    Returns:
        _type_: _description_
    """
    groups: List[Tuple[str, ...]]
    _lut: Dict[str, str] = field(init=False, repr=False)
    __hash__ = object.__hash__

    def __post_init__(self) -> None:
        lut: Dict[str, str] = {}
        for group in self.groups:
            if not group:
                continue
            canonical = group[0]
            for alias in group:
                key = _norm(alias)
                if key in lut and lut[key] != canonical:
                    raise argparse.ArgumentTypeError(
                        f"Alias {alias!r} maps to multiple canonicals: {lut[key]!r} and {canonical!r}"
                    )
                lut[key] = canonical
        self._lut = lut
        self.info = {}

    def __call__(self, value: str) -> str:
        try:
            return self._lut[_norm(value)]
        except KeyError:
            raise argparse.ArgumentTypeError(self.error_message(value))

    @property
    def canonicals(self) -> List[str]:
        return [g[0] for g in self.groups if g]

    def aliases(self, canon:str) -> List[str]:
        return next((g[1:] for g in self.groups if g[0]==canon and len(g)>1),[])

    def _info_help_single(self, canon: str):
        als = self.aliases(canon)
        return f"{canon} ({', '.join(als)})" if als else canon
    
    def _info_help(self, groups: Optional[List[Tuple[str, ...]]] = None) -> str:
        groups = groups or self.groups
        parts = []
        for g in groups:
            canon, *als = g
            parts.append(f"{canon} ({', '.join(als)})" if als else canon)
        return "; ".join(parts)

    def help_info(self, info: dict):
        parts = ['Options:']
        for g in self.groups:
            canon, *als = g
            g_info = info.get(canon,'')
            info_str = f'{canon}: {g_info}'
            if als:
                info_str += f'. Accepted aliases: {', '.join(als)}'
            parts.append(info_str)
        return '\n'.join(parts)

    def help_suffix(self, info: Optional[dict]=None) -> str:
        info = info or self.info
        if info:
            return self.help_info(info)
        
        return "Aliases: " + self._info_help() + "."

    def error_message(self, bad: str) -> str:
        return (f"Invalid choice: {bad!r}. "
                f"Accepted choices (choice aliases): {self._info_help()}.")

def make_aliases_from_dict(d: dict, key:Optional[str]='', info: Optional[dict]=None) -> Aliases:
    info = info or {}
    if key and key in d.keys():
        group_d = d.get(key)
    else:
        group_d = d
    if not info:
        for k,v in group_d.items():
            if isinstance(v,dict):
                info[k] = v.get('info','')
    AL = base_make_aliases_from_dict(group_d)
    AL.info = info
    return AL
        
def base_make_aliases_from_dict(d: dict, key:Optional[str]='') -> Aliases:
    def make_tuple(k,v):
        if isinstance(v, dict):
            v = v.get('aliases',[])
        if v:
            return (k, *v)
        else:
            return (k,)
    if key and key in d.keys():
        groups = [make_tuple(k,v) for k,v in d.get(key).items()]
    else:
        groups = [make_tuple(k,v) for k,v in d.items()]
    return make_aliases(*groups)

def make_aliases(*groups: Iterable[str]) -> Aliases:
    """
    Convenience builder: each group is an iterable where the first element
    is the canonical spelling, followed by any aliases.
    """
    # Normalize to tuples
    tgroups = [tuple(g) for g in groups]
    return Aliases(tgroups)

# --------------------
# Helpers for list inputs
# --------------------
def _dedup_keep_order(items: List[str]) -> List[str]:
    seen = set(); out = []
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def parse_list_value(value: Any, OPTIONS: Aliases) -> List[str]:
    """
    Accept str (CSV/space), list[str], or tuple[str]; map each token via OPTIONS.
    """
    html_commas = ['%252C','%2C']
    for comma in html_commas:
        if comma in value:
            value = value.replace(comma,',')
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens = []
        for v in value:
            tokens += [t for t in re.split(r"[,\s]+", str(v)) if t]
    else:
        tokens = [t for t in re.split(r"[,\s]+", str(value)) if t]

    out = []
    for tok in tokens:
        canon = OPTIONS(tok)  # will raise on bad token
        if canon != "default":  # ignore 'default' in plural form
            out.append(canon)
    return _dedup_keep_order(out)

