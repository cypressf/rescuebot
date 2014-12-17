from math import pi, radians

TAU = 2 * pi


def ranges_to_points(ranges):
    points = []
    for angle, length in enumerate(ranges):
        points.append(LaserPoint(length=length, angle=angle))
    return points


def filter_points(points):
    return [point for point in points if point.length]


class LaserPoint:
    def __init__(self, length=0.0, angle=0.0):
        self.length = length  # meters
        self.angle_degrees = angle
        self.angle_radians = radians(angle)

    def __lt__(self, other):
        return self.length < other.length

    def __str__(self):
        return "length: %.2f  angle: %.3f" % (self.length, self.angle_radians / TAU)

    def is_in_front(self):
        return self.is_in_front_right() or self.is_in_front_left()

    def is_in_front_left(self):
        return 0 < self.angle_radians <= TAU * 1 / 4.0

    def is_in_back_left(self):
        return TAU * 1 / 4.0 < self.angle_radians <= TAU * 1 / 2.0

    def is_in_back_right(self):
        return TAU * 1 / 2.0 < self.angle_radians <= TAU * 3 / 4.0

    def is_in_front_right(self):
        return TAU * 3 / 4.0 < self.angle_radians <= TAU

