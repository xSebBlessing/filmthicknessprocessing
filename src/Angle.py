# trigonometry
import math


class Angle(object):
    """
    This class stores angles in deg and rad for easier handling in equations and user output
    """
    def __init__(self, angle: float, unit: str = "deg"):
        self._deg: float = 0.0
        self._rad: float = 0.0

        if unit == "deg":
            self._deg = angle
            self._rad = angle / 180 * math.pi
        elif unit == "rad":
            self._deg = angle * 180 / math.pi
            self._rad = angle
        else:
            print('Unknown angle unit name! Only deg or rad available.')
            exit(-1)

    def deg(self) -> float:
        return self._deg

    def rad(self) -> float:
        return self._rad
