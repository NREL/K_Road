from math import *


def clamp(value, minimum=-1.0, maximum=1.0):
    return max(minimum, min(maximum, value))


def scale_and_clamp(value, scale, minimum=-1.0, maximum=1.0):
    return clamp(value / scale, minimum, maximum)


def hinge_transform(value, minimum, maximum):
    return value * maximum if value >= 0.0 else value * -minimum


def reset_angle(angle):
    """
    :param angle: an angle in radians
    :return: aliased angle within the range -pi to +pi
    """
    return (angle + pi) % tau - pi


def delta_angle(a, b):
    """
    :param a: an angle in radians
    :param b: another angle in radians
    :return: The smallest angle between angles a and b, expressed in the range -pi to +pi.
    """
    return reset_angle(a - b)


def signed_delta_angle(a: float, b: float) -> float:
    """
    :param a: an angle in radians
    :param b: another angle in radians
    :return: The signed smallest angle between angles a and b, expressed in the range -pi to +pi, using the RHR
    """
    r1 = (a - b) % tau
    r2 = (b - a) % tau
    return r1 if r1 <= r2 else -r2


def dampen_val(val, lim, coef):
    damped = val * coef
    if fabs(damped) < lim:
        return 0.0
    else:
        return damped


def dampen_val(val, lim, coef):
    damped = val * coef
    if fabs(damped) < lim:
        return 0.0
    else:
        return damped


def exponential_transform(gain, x):
    y = (x + 1) / 2
    return 2 * (exp(gain * y) - 1) / (exp(gain) - 1) - 1


def two_sided_smooth_exponential(gain, x):
    return exp(gain * (x - 1)) - exp(-gain * (x + 1))


def two_sided_exponential(gain, x):
    return copysign(1, x) * (exp(gain * (fabs(x) - 1)) - exp(-gain)) / (1 - exp(-gain))


def inverse_two_sided_exponential(gain, y):
    z = fabs(y)
    return copysign(1, y) * log(z * exp(gain) - z + 1) / gain


def two_sided_offset_exponential(gain, offset, x):
    y = 0
    s = 1
    if x < offset:
        y = 1 - (x + 1) / (offset + 1)
        s = -1
    else:
        y = (x - offset) / (1 - offset)
    return s * (exp(gain * y) - 1) / (exp(gain) - 1)


def linear_offset(offset, x):
    if x < offset:
        return (x + 1) / (offset + 1) - 1
    else:
        return (x - offset) / (1 - offset)


def linear_offset_exponential(gain, offset, x):
    if offset > 0:
        if x < offset:
            return linear_offset(offset, x)
        else:
            return two_sided_offset_exponential(gain, offset, x)
    else:
        if x < offset:
            return two_sided_offset_exponential(gain, offset, x)
        else:
            return linear_offset(offset, x)
