import numpy as np
import math

class Kinematics:
    def __init__(self, velocity, angle, type = 'line',):
        self.type = type
        self.velocity = velocity
        self.angle = angle



def modifiy_position(pos, xlim, ylim, kine):
    # modify the position pox = (x,y) at step t
    # according to a config kine (kinematics) and limits of the grid (xlim/ylim)

    if kine.type == 'line':
        x,y = pos
        x += kine.velocity * math.cos(kine.angle)
        y += kine.velocity * math.sin(kine.angle)

        if x<xlim[0] or x>xlim[1] or y<ylim[0] or y>ylim[1]:
            # if new position is outside the limits
            # we change angle
            x -= 2*kine.velocity * math.cos(kine.angle)
            y -= 2*kine.velocity * math.sin(kine.angle)

            kine.angle += math.pi # change direction

    return [x,y] , kine


