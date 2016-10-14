import imagen as ig
import numpy as np
import scipy.io as sio
import itertools
import perturbation as pert
from holoviews.core import BoundingBox
import param

from imagen.patterngenerator import PatternGenerator, Composite
from imagen.patternfn import float_error_ignore


# ##################### SineGrating
def SineG(x, y, BlackBackground, jetter, Bound):
    # creates parameter list
    orientations = [i * np.pi / 12 for i in range(12)]
    frequencies = [3, 6, 12]
    of = list(itertools.product(orientations, frequencies))
    if jetter:
        of = pert.modulation_of_jetter(of,'SG')

    # generates images and saved into a numpy array
    SG = np.empty(len(of), dtype=np.object)
    idx = 0
    for ith_of in of:
        SG[idx] = ig.SineGrating(bounds=Bound, phase=np.pi / 2, orientation=ith_of[0],
                                  frequency=ith_of[1], xdensity=x, ydensity=y)()
        # SG[idx] = ig.SineGrating(bounds=Bound, phase=0, orientation=ith_of[0],
        #                           frequency=ith_of[1], xdensity=x, ydensity=y)()

        if BlackBackground:
            assert SG[idx].max() <= 1.0 and SG[idx].min() >= 0
            SG[idx] = 1 - SG[idx]
        idx += 1
    return SG


class HyperbolicGrating(PatternGenerator):
    """
    Concentric rectangular hyperbolas with Gaussian fall-off which share the same asymptotes.
    abs(x^2/a^2 - y^2/a^2) = 1, where a mod size = 0
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    thickness = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness of the hyperbolas.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the hyperbolas.")

    size = param.Number(default=0.5,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.62,doc="Size as distance of inner hyperbola vertices from the centre.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = p.thickness
        gaussian_width = p.smoothing
        size = p.size

        distance_from_vertex_middle = np.fmod(np.sqrt(np.absolute(x**2 - y**2)),size)
        distance_from_vertex_middle = np.minimum(distance_from_vertex_middle,size - distance_from_vertex_middle)

        return (np.cos(distance_from_vertex_middle/size*2*np.pi)+1)/2


# ##################### HyperbolicGrating
def HyperG(x, y, BlackBackground, jetter, Bound):

    orientations = [i*np.pi/8 for i in range(4)]
    sizes = [1.0/3,1.0/6,1.0/12]
    thickness = [1.0/30,1.0/60,1.0/120]
    smoothing = [1.0/15,1.0/30,1.0/60]
    s_t_m = zip(sizes,thickness,smoothing)
    paras = [(o,s,t,m) for o in orientations for s,t,m in s_t_m]

    if jetter:
        paras = pert.modulation_of_jetter(paras,'HG')

    # generates images and saved into a numpy array
    HG = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        HG[idx] = HyperbolicGrating(bounds=Bound, thickness=par[2], size=par[1],
                                        orientation=par[0], smoothing=par[3], xdensity=x, ydensity=y)()
        if BlackBackground:
            assert HG[idx].max() <= 1.0 and HG[idx].min() >= 0
            HG[idx] = 1 - HG[idx]
        idx += 1

    return HG


class Spiral(PatternGenerator):
    """
    Archimedean spiral.
    Successive turnings of the spiral have a constant separation distance.

    Spiral is defined by polar equation r=size*angle plotted in Gaussian plane.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    thickness = param.Number(default=0.02,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the spiral.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the spiral.")

    turning = param.Number(default=0.05,bounds=(0.01,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = p.thickness
        gaussian_width = p.smoothing
        turning = p.turning

        spacing = turning*2*np.pi

        distance_from_origin = np.sqrt(x**2+y**2)
        distance_from_spiral_middle = np.fmod(spacing + distance_from_origin - turning*np.arctan2(y,x),spacing)

        distance_from_spiral_middle = np.minimum(distance_from_spiral_middle,spacing - distance_from_spiral_middle)
        return (np.cos(distance_from_spiral_middle/spacing*2*np.pi)+1)/2
#         distance_from_spiral = distance_from_spiral_middle - thickness/2.0
#
#         spiral = 1.0 - np.greater_equal(distance_from_spiral,0.0)
#
#         sigmasq = gaussian_width*gaussian_width
#
#         with float_error_ignore():
#             falloff = np.exp(np.divide(-distance_from_spiral*distance_from_spiral, 2.0*sigmasq))
#
#         return np.maximum(falloff, spiral)

class SpiralGrating(Composite):
    """
    Grating pattern made from overlaid spirals.
    """

    parts = param.Integer(default=2,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    thickness = param.Number(default=0.00,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the spiral.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the spiral.")

    turning = param.Number(default=0.05,bounds=(0.01,None),softbounds=(0.01,2.0),
        precedence=0.62,doc="Density of turnings; turning*angle gives the actual radius.")


    def function(self, p):
        o=2*np.pi/p.parts
        gens = [Spiral(turning=p.turning,smoothing=p.smoothing,thickness=p.thickness,
                       orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()

# ##################### SpiralGrating
def SpiralG(x, y, BlackBackground, jetter, Bound):
    turning = [0.2,0.1,0.05]
    turning_inc = [0.3,0.15,0.075]
    smooth = [0.16,0.08,0.04]
    smooth_inc = [0.018,0.009,0.0045]
    t_s = zip(turning,turning_inc,smooth,smooth_inc)
    parts = [2, 4, 6, 8]
    t = [p[0]+p[1]*(part/2-1) for p in t_s for part in parts]
    s = [p[2]+p[3]*(part/2-1) for p in t_s for part in parts]

    # turning = [1.0/3,1.0/6,1.0/12]
    # smooth = [0.16,0.08,0.04]
    # parts = [2, 4, 6, 8]
    # t = [p/part for p in turning for part in parts]
    # s = [p/part for p in smooth for part in parts]

    parts_m = [part for p in turning for part in parts]
    paras = zip([0.0]*len(t),t,s,parts_m)

    if jetter:
        paras = pert.modulation_of_jetter(paras,'SpG')

    SpG = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        SpG[idx] = ig.SpiralGrating(bounds=Bound,parts=par[3],turning=par[1],smoothing=par[2],
                                    orientation=par[0],xdensity=x,ydensity=y)()
        if BlackBackground:
            assert SpG[idx].max() <= 1.0 and SpG[idx].min() >= 0
            SpG[idx] = 1-SpG[idx]
        idx += 1

    return SpG

class Wedge(PatternGenerator):
    """
    A sector of a circle with Gaussian fall-off, with size determining the arc length.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    size = param.Number(default=np.pi/4,bounds=(0.0,None),softbounds=(0.0,2.0*np.pi),
        precedence=0.60,doc="Angular length of the sector, in radians.")

    smoothing = param.Number(default=0.4,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off outside the sector.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        gaussian_width = p.smoothing

        angle = np.absolute(np.arctan2(y,x))
        half_length = p.size/2

        radius = 1.0 - np.greater_equal(angle,half_length)
        distance = angle - half_length

        sigmasq = gaussian_width*gaussian_width

        with float_error_ignore():
            falloff = np.exp(np.divide(-distance*distance, 2.0*sigmasq))

        return np.maximum(radius, falloff)
 #       return (np.cos(angle/half_length*np.pi)+1)/2

class RadialGrating(Composite):
    """
    Grating pattern made from alternating smooth circular segments (pie-shapes).
    """

    parts = param.Integer(default=4,bounds=(1,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Number of parts in the grating.")

    smoothing = param.Number(default=0.8,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="""
        Width of the Gaussian fall-off outside the sector, scaled by parts.""")

    def function(self, p):
        o=2*np.pi/p.parts
        gens = [Wedge(size=np.pi/p.parts*2/3,smoothing=p.smoothing/p.parts,
                      orientation=i*2*np.pi/p.parts) for i in range(p.parts)]

        return Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                         xdensity=p.xdensity, ydensity=p.ydensity)()


# ##################### RadialGrating
def RadialG(x, y, BlackBackground, jetter, Bound):
    parts = [1]*4+[2,4,8,16]
    orien = [np.pi/2*j for j in range(4)]+ [0]*4
    smooth = [0.7]*4 + [0.7]*4

    paras = zip(orien,smooth,parts)

    if jetter:
        paras = pert.modulation_of_jetter(paras,'RG')

    RG = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        if par[2] == 1:
            RG[idx] = ig.RadialGrating(bounds=Bound,parts=par[2],smoothing=par[1],
                                   orientation=par[0],xdensity=x,ydensity=y)()
        else:
            RG[idx] = RadialGrating(bounds=Bound,parts=par[2],smoothing=par[1],
                                   orientation=par[0],xdensity=x,ydensity=y)()
        if BlackBackground:
            assert RG[idx].max() <= 1.0 and RG[idx].min() >= 0
            RG[idx] = 1-RG[idx]
        idx += 1

    return RG

class ConcentricRings(PatternGenerator):
    """
    Concentric rings with linearly increasing radius.
    Gaussian fall-off at the edges.
    """

    aspect_ratio = param.Number(default=1.0,bounds=(0.0,None),softbounds=(0.0,2.0),
        precedence=0.31,doc="Ratio of width to height.")

    thickness = param.Number(default=0.04,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.60,doc="Thickness (line width) of the ring.")

    smoothing = param.Number(default=0.05,bounds=(0.0,None),softbounds=(0.0,0.5),
        precedence=0.61,doc="Width of the Gaussian fall-off inside and outside the rings.")

    size = param.Number(default=0.4,bounds=(0.01,None),softbounds=(0.1,2.0),
        precedence=0.62,doc="Radius difference of neighbouring rings.")

    def function(self,p):
        aspect_ratio = p.aspect_ratio
        x = self.pattern_x/aspect_ratio
        y = self.pattern_y
        thickness = p.thickness
        gaussian_width = p.smoothing
        size = p.size

        distance_from_origin = np.sqrt(x**2+y**2)

        distance_from_ring_middle = np.fmod(distance_from_origin,size)
        distance_from_ring_middle = np.minimum(distance_from_ring_middle,size - distance_from_ring_middle)
        return (np.cos(distance_from_ring_middle/size*2*np.pi)+1)/2

# #################### Targets/ ConcentricRings
def Targets(x, y, BlackBackground, jetter, Bound):
    sizes = [1.0/3, 1.0/6, 1.0/12]
    thickness = [1.0/30,1.0/60,1.0/120]
    smoothing = [1.0/15,1.0/30,1.0/60]
    paras = zip([0]*3,sizes,thickness,smoothing,[1]*3)

    if jetter:
        paras = pert.modulation_of_jetter(paras,'T')

    T = np.empty(len(paras), dtype=np.object)
    idx = 0
    for par in paras:
        T[idx] = ConcentricRings(bounds=Bound, aspect_ratio=par[4], smoothing=par[3], thickness=par[2],
                                    size=par[1], orientation=par[0], xdensity=x, ydensity=y)()
        if BlackBackground:
            assert T[idx].max() <= 1.0 and T[idx].min() >= 0
            T[idx] = 1 - T[idx]
        idx += 1

    return T



# ################### Bars
def Bar(x,y,BlackBackground,jetter,Bound):
    orientations = [i*np.pi/18 for i in range(18)]
    sizes = [0.5,0.25]
    paras = list(itertools.product(orientations,sizes))
    if jetter:
        paras = pert.modulation_of_jetter(paras,'Bar')

    Ba = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        Ba[idx] = ig.Rectangle(bounds=Bound,smoothing=0.015,aspect_ratio=0.1,size=par[1],orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert Ba[idx].max() <= 1.0 and Ba[idx].min >= 0
            Ba[idx] = 1-Ba[idx]
        idx += 1

    return Ba


if __name__ == '__main__':
    # image size
    # x = 256
    # y = 256
    # Bound = BoundingBox(radius=1.16)
    x = 256
    y = 256
    Bound = BoundingBox(radius=0.58*2)
    BlackBackground = True  # convert image to black background
    jetter = False  # add perturbations


# ############# waves
    SG = SineG(x, y, BlackBackground, jetter, Bound)
    HG = HyperG(x, y, BlackBackground, jetter, Bound)
    SpG = SpiralG(x, y, BlackBackground, jetter, Bound)
    RG = RadialG(x, y, BlackBackground, jetter, Bound)
    T = Targets(x, y, BlackBackground, jetter, Bound)

    # ############## save images to .mat file
    if jetter:
        if BlackBackground:
            sio.savemat('SG_pert_b.mat', {'imgs': SG})
            sio.savemat('HG_pert_b.mat', {'imgs': HG})
            sio.savemat('SpG_pert_b.mat', {'imgs': SpG})
            sio.savemat('RG_pert_b.mat', {'imgs': RG})
            sio.savemat('T_pert_b.mat', {'imgs': T})
        else:
            sio.savemat('SG_pert_w.mat', {'imgs': SG})
            sio.savemat('HG_pert_w.mat', {'imgs': HG})
            sio.savemat('SpG_pert_w.mat', {'imgs': SpG})
            sio.savemat('RG_pert_w.mat', {'imgs': RG})
            sio.savemat('T_pert_w.mat', {'imgs': T})
    else:
        if BlackBackground:
            sio.savemat('SG_b.mat', {'imgs': SG})
            sio.savemat('HG_b.mat', {'imgs': HG})
            sio.savemat('SpG_b.mat', {'imgs': SpG})
            sio.savemat('RG_b.mat', {'imgs': RG})
            sio.savemat('T_b.mat', {'imgs': T})
        else:
            sio.savemat('SG_w.mat', {'imgs': SG})
            sio.savemat('HG_w.mat', {'imgs': HG})
            sio.savemat('SpG_w.mat', {'imgs': SpG})
            sio.savemat('RG_w.mat', {'imgs': RG})
            sio.savemat('T_w.mat', {'imgs': T})



# # ##########primitives
#     B = Bar(x, y, BlackBackground, jetter, Bound)
#     if jetter:
#         if BlackBackground:
#             sio.savemat('B_pert_b.mat', {'imgs': B})
#         else:
#             sio.savemat('B_pert_w.mat', {'imgs': B})
#     else:
#         if BlackBackground:
#             sio.savemat('B_b.mat', {'imgs': B})
#         else:
#             sio.savemat('B_w.mat', {'imgs': B})
