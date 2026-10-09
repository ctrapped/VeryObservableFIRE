import ast
import math
import os
import numpy as np

####Reads a plain-text "key = value" parameter file into a dict, instead of importing a .py module.
####Values are parsed top-to-bottom and may reference earlier keys in the same file, e.g.:
####    galName = m12m
####    fileDir = /path/to/sims/{galName}/snapdir_      #string templating
####    bandwidth_km_s = 256*res_km_s                    #arithmetic expression (any earlier key may be used this way)
####`pi` and `arcsec` are always available in expressions without needing to be defined in the file.
####
####Written By Cameron Trapp (ctrapped@gmail.com)

pi = math.pi
arcsec = (1. / 60. / 60.) * pi / 180.

_CONSTANTS = {"pi": pi, "arcsec": arcsec}

#These are wrapped in np.array(...) after parsing to match prior behavior, when not overridden
_ARRAY_FIELDS = ("observerVelocity", "inclinations")


def _strip_comment(line):
    #Remove a trailing '# comment', ignoring '#' that appears inside a quoted string
    in_quote = None
    for i, ch in enumerate(line):
        if in_quote:
            if ch == in_quote:
                in_quote = None
        elif ch in ("'", '"'):
            in_quote = ch
        elif ch == '#':
            return line[:i]
    return line


def _parse_value(raw, namespace):
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        pass
    try:
        return eval(raw, {"__builtins__": {}}, namespace)
    except Exception:
        pass
    try:
        return raw.format(**namespace)
    except (KeyError, IndexError):
        return raw


def LoadParamFile(path, overrides=None):
    #Parse the file, applying any given overrides inline (so later lines that reference
    #an overridden key, e.g. a path using "{galName}", see the overridden value)
    overrides = overrides or {}
    params = {}
    namespace = dict(_CONSTANTS)

    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = _strip_comment(line).strip()
            if not line:
                continue
            if '=' not in line:
                raise ValueError(f"{path}:{lineno}: expected 'key = value', got: {line!r}")

            key, _, raw_value = line.partition('=')
            key = key.strip()

            if key in overrides and overrides[key] is not None:
                value = overrides[key]
            else:
                value = _parse_value(raw_value, namespace)

            params[key] = value
            namespace[key] = value

    #Apply any overrides for keys the file didn't actually define
    for key, value in overrides.items():
        if value is not None and key not in params:
            params[key] = value

    return params


def LoadParams(path, galName=None, minSnap=None, maxSnap=None, inclination=None):
    #Read a parameter file, applying the same overrides VeryObservableFIRE.py's
    #command-line arguments previously applied via LoadFileInfo/LoadObserverInfo.
    if not os.path.isfile(path) and os.path.isfile(path + ".param"):
        path = path + ".param"

    params = LoadParamFile(path, overrides={"galName": galName, "minSnap": minSnap, "maxSnap": maxSnap})

    for field in _ARRAY_FIELDS:
        if field in params and not isinstance(params[field], np.ndarray):
            params[field] = np.array(params[field])

    if inclination is not None:
        params["inclinations"] = [inclination] #Matches prior behavior: overriding to a single inclination used a plain list, not np.array

    return params
