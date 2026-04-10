from math import exp, log
from matplotlib.colors import TwoSlopeNorm, AsinhNorm, CenteredNorm

NORMFUNCS = {}

def register(name):
    def decorator(func):
        NORMFUNCS[name] = func
        return func

    return decorator


# ------------------------------------------------------------------------------


def nlog(x, a):
    if a <= 0:
        raise ValueError("Expected a>0 but got a=%f" % a)
    return log(a * x + 1.0) / log(a + 1.0)


# ------------------------------------------------------------------------------
def nexp(x, a):
    if a <= 0:
        raise ValueError("Expected a>0 but got a=%f" % a)
    return (exp(x * log(a + 1.0)) - 1.0) / a


# ------------------------------------------------------------------------------


@register("log_scale")
def log_scale(x):
    return nlog(x, 10.0)


# ------------------------------------------------------------------------------


def log_scale1(x):
    return nlog(x, 1.0)


# ------------------------------------------------------------------------------


@register("exp_scale")
def exp_scale(x):
    return nexp(x, 10.0)


# ------------------------------------------------------------------------------


def exp_scale20(x):
    return nexp(x, 20.0)


# ------------------------------------------------------------------------------


def exp_scale30(x):
    return nexp(x, 30.0)

@register("TwoSlopeNorm")
def twoslopenorm(x):
    return TwoSlopeNorm(0.9, vmin=0, vmax=1)(x)

@register("AsinhNorm")
def asinhnorm(x):
    return AsinhNorm(linear_width=1, vmin=0, vmax=1)(x)

@register("CenteredNorm")
def centerednorm(x):
    print(x, CenteredNorm(vcenter=0.3, halfrange=0.3)(x))
    return CenteredNorm(vcenter=0.3, halfrange=0.3)(x)
