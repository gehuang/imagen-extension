import imagen as ig
import numpy as np
import scipy.io as sio
import perturbation as pert
from holoviews.core import BoundingBox
import param

from imagen.patterngenerator import PatternGenerator, Composite
from imagen.patternfn import float_error_ignore


# ##################### SineGrating
def SineG(x, y, BlackBackground, phase_shift, Bound,frames):

    orientations = [i * np.pi /8 for i in range(8)]
    frequencies = [1.0 ,2.0, 4.0, 8.0, 12.0, 16.0]
    phase = np.pi/2
    paras = [[[ori, fre, phase]] for ori in orientations for fre in frequencies]

    if phase_shift:
        paras = pert.modulation_of_phase(paras,'SG',frames)
    else:
        frames = 1

    # generates images and saved into a numpy array
    SG = np.empty([len(paras), frames], dtype=np.object)
    idx_proto = 0
    for pars in paras:
        idx_shift = 0
        for p in pars:
            SG[idx_proto][idx_shift] = ig.SineGrating(bounds=Bound, phase=p[2], orientation=p[0],
                                      frequency=p[1], xdensity=x, ydensity=y)()

            if BlackBackground:
                assert SG[idx_proto][idx_shift].max() <= 1.0 and SG[idx_proto][idx_shift].min() >= 0
                SG[idx_proto][idx_shift] = 1 - SG[idx_proto][idx_shift]
            idx_shift += 1
        idx_proto += 1
    return SG




# ##################### HyperbolicGrating
class HyperbolicGrating(PatternGenerator):
    """
    Concentric rectangular hyperbolas with Gaussian fall-off which share the same asymptotes.
    abs(x^2/a^2 - y^2/a^2) = 1, where a mod size = 0
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    size = param.Number(default=0.5,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.62,doc="Size as distance of inner hyperbola vertices from the centre.")

    phase = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,2.0*np.pi),
        precedence=0.51,doc="Phase shift from the centre.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        size = p.size
        phase = p.phase
        distance_from_vertex_middle = np.fmod(np.sqrt(np.absolute(x**2 - y**2))+phase*size/(2*np.pi),size)
        distance_from_vertex_middle = np.minimum(distance_from_vertex_middle,size - distance_from_vertex_middle)

        return (np.cos(distance_from_vertex_middle/size*2*np.pi)+1)/2

def HyperG(x, y, BlackBackground, phase_shift, Bound,frames):

    orientations = [i*np.pi/8 for i in range(4)]
    sizes = [1.0/s for s in [1 ,2, 4, 8, 12, 16]]
    paras = [[[o,s,0.0]] for o in orientations for s in sizes]

    if phase_shift:
        paras = pert.modulation_of_phase(paras,'HG',frames)
    else:
        frames = 1

    HG = np.empty([len(paras),frames],dtype=np.object)
    idx_proto = 0
    for pars in paras:
        idx_shift = 0
        for p in pars:
            HG[idx_proto][idx_shift] = HyperbolicGrating(bounds=Bound, orientation=p[0],size=p[1],
                                            phase=p[2], xdensity=x, ydensity=y)()
            if BlackBackground:
                assert HG[idx_proto][idx_shift].max() <= 1.0 and HG[idx_proto][idx_shift].min() >= 0
                HG[idx_proto][idx_shift] = 1 - HG[idx_proto][idx_shift]
            idx_shift += 1
        idx_proto += 1
    return HG




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


    parts = param.Integer(default=2,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    turning = param.Number(default=0.05,bounds=(0.001,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")


    def function(self, p):
        o=2*np.pi/p.parts
        gens = [Spiral(turning=p.turning,outer_parts=p.parts,
                       orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()

def SpiralG(x, y, BlackBackground, phase_shift, Bound, frames):
    parts = [1, 2, 4, 8, 12, 16]
    C_target = [1.0 ,2.0, 4.0, 8.0, 12.0, 16.0]
    paras = [[[np.pi/2, part/(C_t*2*np.pi), part]] for part in parts for C_t in C_target]

    if phase_shift:
        paras = pert.modulation_of_phase(paras,'SpG',frames)
    else:
        frames = 1

    SpG = np.empty([len(paras),frames],dtype=np.object)
    idx_proto = 0
    for pars in paras:
        idx_shift = 0
        for p in pars:
            SpG[idx_proto][idx_shift] = SpiralGrating(bounds=Bound,parts=p[2],turning=p[1],
                                        orientation=p[0],xdensity=x,ydensity=y)()
            if BlackBackground:
                assert SpG[idx_proto][idx_shift].max() <= 1.0 and SpG[idx_proto][idx_shift].min() >= 0
                SpG[idx_proto][idx_shift] = 1-SpG[idx_proto][idx_shift]
            idx_shift += 1
        idx_proto += 1

    return SpG




# ##################### RadialGrating
class Wedge(PatternGenerator):

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    outer_parts = param.Integer(default=4,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y

        thickness = np.pi/p.outer_parts
        angle = np.absolute(np.arctan2(y,x))
        my_wedge = (np.cos(np.pi*angle/thickness)+1)/2
        get_index = angle > thickness
        my_wedge[get_index] = 0
        return my_wedge

class RadialGrating(Composite):

    parts = param.Integer(default=4,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    def function(self, p):
        gens = [Wedge(outer_parts=p.parts,orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()

def RadialG(x, y, BlackBackground, phase_shift, Bound,frames):
    parts = [1,2,4,8,12,16]
    orien = [np.pi/2]*6
    paras = zip(orien,parts)
    paras = [[par] for par in paras]

    if phase_shift:
        paras = pert.modulation_of_phase(paras,'RG',frames)
    else:
        frames = 1

    RG = np.empty([len(paras),frames],dtype=np.object)
    idx_proto = 0
    for pars in paras:
        idx_shift = 0
        for p in pars:
            RG[idx_proto][idx_shift] = RadialGrating(bounds=Bound,parts=p[1],
                                       orientation=p[0],xdensity=x,ydensity=y)()
            if BlackBackground:
                assert RG[idx_proto][idx_shift].max() <= 1.0 and RG[idx_proto][idx_shift].min() >= 0
                RG[idx_proto][idx_shift] = 1-RG[idx_proto][idx_shift]
            idx_shift += 1
        idx_proto += 1

    return RG




class ConcentricRings(PatternGenerator):
    """
    Concentric rings with linearly increasing radius.
    Gaussian fall-off at the edges.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    size = param.Number(default=0.4,bounds=(0.01,None),softbounds=(0.1,2.0),
        precedence=0.62,doc="Radius difference of neighbouring rings.")

    phase = param.Number(default=0.0,bounds=(0.0,None),softbounds=(0.0,2.0*np.pi),
        precedence=0.51,doc="Phase shift from the centre.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        size = p.size
        phase = p.phase
        distance_from_origin = np.sqrt(x**2+y**2)

        distance_from_ring_middle = np.fmod(distance_from_origin + phase*size/(2*np.pi),size)
        distance_from_ring_middle = np.minimum(distance_from_ring_middle,size - distance_from_ring_middle)
        return (np.cos(distance_from_ring_middle/size*2*np.pi)+1)/2

# #################### Targets/ ConcentricRings
def Targets(x, y, BlackBackground, phase_shift, Bound,frames):
    # sizes = [1.0/s for s in [1.2 ,2.4, 4.8, 9.6, 13.6, 19.2,27.2]]
    sizes = [1.0/s for s in [1 ,2, 4, 8, 12, 16]]
    paras = [[[0, s, 1, 0.0]] for s in sizes]

    if phase_shift:
        paras = pert.modulation_of_phase(paras,'T',frames)
    else:
        frames = 1

    T = np.empty([len(paras),frames], dtype=np.object)
    idx_proto = 0
    for pars in paras:
        idx_shift = 0
        for p in pars:
            T[idx_proto][idx_shift] = ConcentricRings(bounds=Bound, aspect_ratio=p[2],
                                        size=p[1], orientation=p[0], phase=p[3], xdensity=x, ydensity=y)()
            if BlackBackground:
                assert T[idx_proto][idx_shift].max() <= 1.0 and T[idx_proto][idx_shift].min() >= 0
                T[idx_proto][idx_shift] = 1 - T[idx_proto][idx_shift]
            idx_shift += 1
        idx_proto += 1

    return T



if __name__ == '__main__':
    # image size
    x = 800/1.2
    y = 800/1.2
    Bound = BoundingBox(radius=0.6)
    BlackBackground = True  # convert image to black background
    phase_shift = False  # generate different phases
    f = 4

    SpG = SpiralG(x, y, BlackBackground, phase_shift, Bound, frames=f)
    sio.savemat('SpG_b.mat', {'imgs': SpG})

    # ############ waves
    # SG = SineG(x, y, BlackBackground, phase_shift, Bound, frames=f)
    # SpG = SpiralG(x, y, BlackBackground, phase_shift, Bound, frames=f)
    # T = Targets(x, y, BlackBackground, phase_shift, Bound, frames=f)
    # HG = HyperG(x, y, BlackBackground, phase_shift, Bound, frames=f)
    # RG = RadialG(x, y, BlackBackground, phase_shift, Bound, frames=f)
    #
    # ############## save images to .mat file
    # if phase_shift:
    #     if BlackBackground:
    #         sio.savemat('SG_shift_b.mat', {'imgs': SG})
    #         sio.savemat('HG_shift_b.mat', {'imgs': HG})
    #         sio.savemat('SpG_shift_b.mat', {'imgs': SpG})
    #         sio.savemat('RG_shift_b.mat', {'imgs': RG})
    #         sio.savemat('T_shift_b.mat', {'imgs': T})
    #     else:
    #         sio.savemat('SG_shift_w.mat', {'imgs': SG})
    #         sio.savemat('HG_shift_w.mat', {'imgs': HG})
    #         sio.savemat('SpG_shift_w.mat', {'imgs': SpG})
    #         sio.savemat('RG_shift_w.mat', {'imgs': RG})
    #         sio.savemat('T_shift_w.mat', {'imgs': T})
    # else:
    #     if BlackBackground:
    #         sio.savemat('SG_b.mat', {'imgs': SG})
    #         sio.savemat('HG_b.mat', {'imgs': HG})
    #         sio.savemat('SpG_b.mat', {'imgs': SpG})
    #         sio.savemat('RG_b.mat', {'imgs': RG})
    #         sio.savemat('T_b.mat', {'imgs': T})
    #     else:
    #         sio.savemat('SG_w.mat', {'imgs': SG})
    #         sio.savemat('HG_w.mat', {'imgs': HG})
    #         sio.savemat('SpG_w.mat', {'imgs': SpG})
    #         sio.savemat('RG_w.mat', {'imgs': RG})
    #         sio.savemat('T_w.mat', {'imgs': T})
    #
