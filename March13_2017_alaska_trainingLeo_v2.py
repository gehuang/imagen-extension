# This file was used to generate the 4 outer spiral images

import imagen as ig
import numpy as np
import scipy.io as sio
import perturbation as pert
from holoviews.core import BoundingBox
import param

from imagen.patterngenerator import PatternGenerator, Composite
from imagen.patternfn import float_error_ignore

# This one generates a spiral that each individual part will have different squashed orientations

# ##################### SpiralGrating
class Spiral(PatternGenerator):

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    outer_parts = param.Integer(default=2,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    turning = param.Number(default=0.05,bounds=(0.001,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y

        turning = p.turning

        spacing = turning*2*np.pi
        thickness = float(spacing)/p.outer_parts/2

        distance_from_origin = np.sqrt(x**2+y**2)
        distance_from_spiral_middle = np.fmod(spacing + distance_from_origin - turning*np.arctan2(y,x),spacing)

        distance_from_spiral_middle = np.minimum(distance_from_spiral_middle,spacing - distance_from_spiral_middle)
        get_index = distance_from_spiral_middle > thickness
        my_spiral = (np.cos(distance_from_spiral_middle/thickness*np.pi)+1)/2
        my_spiral[get_index] = 0
        return my_spiral

class SpiralGrating(Composite):

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    parts = param.Integer(default=2,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    turning = param.Number(default=0.05,bounds=(0.001,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")


    def function(self, p):
        o=2*np.pi/p.parts
        gens = [Spiral(turning=p.turning,outer_parts=p.parts,aspect_ratio=p.aspect_ratio,
                       orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()

def SpiralG(x, y, BlackBackground, phase_shift, Bound, frames):
    # parts = [1, 2, 4, 8, 12, 16]
    # C_target = [1.0 ,2.0, 4.0, 8.0, 12.0, 16.0]
    # paras = [[[np.pi/2, part/(C_t*2*np.pi), part]] for part in parts for C_t in C_target]
    C_target = [2.0, 4.0, 12.0, 8.0]
    parts = [8, 2, 4, 6]
    paras = [[[np.pi/2, x[1]/(x[0]*2*np.pi), x[1]]] for x in zip(C_target,parts)]

    scale_idx = [1/1.2, 1.0, 1.2]
    ori = [np.pi*j/4 for j in range(4)]

    if phase_shift:
        paras = pert.modulation_of_phase(paras,'SpG',frames)
    else:
        frames = 12


    SpG = np.empty([len(paras),frames],dtype=np.object)
    idx_proto = 0
    for pars in paras:
        idx_shift = 0
        for p in pars:
            for ith_s in scale_idx:
                for ith_ori in ori:
                    SpG[idx_proto][idx_shift] = SpiralGrating(bounds=Bound,parts=p[2],turning=p[1]*ith_s,aspect_ratio=1.8,
                                                orientation=p[0]+ith_ori,xdensity=x,ydensity=y)()

                    if BlackBackground:
                        assert SpG[idx_proto][idx_shift].max() <= 1.0 and SpG[idx_proto][idx_shift].min() >= 0
                        SpG[idx_proto][idx_shift] = 1-SpG[idx_proto][idx_shift]
                    idx_shift += 1
        idx_proto += 1

    return SpG


if __name__ == '__main__':
    # image size
    x = 192
    y = 192
    Bound = BoundingBox(radius=0.64)
    BlackBackground = True  # convert image to black background
    phase_shift = False  # generate different phases
    f = 4

    ############ waves
    SpG = SpiralG(x, y, BlackBackground, phase_shift, Bound, frames=f)

    ############## save images to .mat file
    if phase_shift:
        if BlackBackground:
            sio.savemat('SpG_shift_b.mat', {'imgs': SpG})
        else:
            sio.savemat('SpG_shift_w.mat', {'imgs': SpG})
    else:
        if BlackBackground:
            sio.savemat('SpG_b.mat', {'imgs': SpG})
        else:
            sio.savemat('SpG_w.mat', {'imgs': SpG})
