import imagen as ig
import numpy as np
import scipy.io as sio
import itertools
import perturbation as pert
from holoviews.core import BoundingBox
import param


class TJunction(ig.Composite):
    """
    T Junction. This is adapted from definition of Asterisk.
    """

    thickness = param.Number(default=0.05, bounds=(0.0, None), softbounds=(0.0, 0.5),
                             precedence=0.60, doc="Thickness of the T junction.")

    smoothing = param.Number(default=0.015, bounds=(0.0, None), softbounds=(0.0, 0.5),
                             precedence=0.61,
                             doc="Width of the Gaussian fall-off around the rectangles composing the junction.")

    size = param.Number(default=0.5, bounds=(0.01, None), softbounds=(0.1, 2.0),
                        precedence=0.62, doc="Overall diameter of the pattern.")

    # generate one long vertical one, and one short (half of long) horizontal one.
    def function(self, p):
        gens = [ig.Rectangle(orientation=0, smoothing=p.smoothing,
                             aspect_ratio=p.thickness / p.size,
                             size=p.size),
                ig.Rectangle(orientation=np.pi / 2, smoothing=p.smoothing,
                             aspect_ratio=2 * p.thickness / p.size,
                             size=p.size / 2, x=p.size / 4)]

        return ig.Composite(generators=gens, bounds=p.bounds, orientation=p.orientation,
                            xdensity=p.xdensity, ydensity=p.ydensity)()



def Bar(x,y,BlackBackground,jetter,Bound):
    orientations = [i*np.pi/18 for i in range(18)]*2
    sizes = [0.5]*18+[0.25]*18
    ratio = [0.1]*18 + [0.2]*18
    paras = zip(orientations,sizes,ratio)

    if jetter:
        paras = pert.modulation(paras,'Bar')

    Ba = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        Ba[idx] = ig.Rectangle(bounds=Bound,smoothing=0.015,aspect_ratio=par[2],size=par[1],orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert Ba[idx].max() <= 1.0 and Ba[idx].min >= 0
            Ba[idx] = 1-Ba[idx]
        idx += 1

    return Ba

def Asterisk(x,y,BlackBackground,jetter,Bound):
    p = [3]*8 + [4]*8 + [5]*4 + [6]*2
    s = [0.5]*4 + [0.25]*4 + [0.5]*4 + [0.25]*4 + [0.5]*2 + [0.25]*2 + [0.5] + [0.25]
    orien = [np.pi/2*i for i in range(4)]+[np.pi/2*i for i in range(4)]+ \
            [np.pi/8*i for i in range(4)] + [np.pi/8*i for i in range(4)] + \
            [np.pi*i for i in range(2)] + [np.pi*i for i in range(2)] + [0] + [0]

    paras = zip(orien, s, p)
    if jetter:
        paras = pert.modulation(paras,'Asterisk')

    Aster = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        Aster[idx] = ig.Asterisk(bounds=Bound,parts=par[2],size=par[1],orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert Aster[idx].max() <= 1.0 and Aster[idx].min >= 0
            Aster[idx] = 1-Aster[idx]
        idx += 1

    return Aster

def Ring(x,y,BlackBackground,jetter,Bound):
    s = [0.5,0.25]
    paras = zip([0]*2,s,[1]*2)

    if jetter:
        paras = pert.modulation(s, 'Ring')

    CR = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        CR[idx] = ig.Ring(bounds=Bound,smoothing=0.015,aspect_ratio=par[2],size=par[1],thickness=0.05, orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert CR[idx].max() <= 1.0 and CR[idx].min >= 0
            CR[idx] = 1-CR[idx]
        idx += 1
    return CR

def Angle(x,y,BlackBackground,jetter,Bound):
    orien = [i*np.pi/2 for i in range(4)]*2 + [i*np.pi/4 for i in range(8)]*2 + [i*np.pi/2 for i in range(4)]*2
    s = [0.5]*4+[0.25]*4+[0.5]*8+[0.25]*8+[0.5]*4+[0.25]*4
    a = [np.pi/8]*8 + [np.pi/4]*16 + [np.pi/3]*8

    paras = zip(orien, s, a)
    if jetter:
        paras = pert.modulation(paras,'Angle')

    An = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        An[idx] = ig.Angle(bounds=Bound,angle=par[2],size=par[1],orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert An[idx].max() <= 1.0 and An[idx].min >= 0
            An[idx] = 1-An[idx]
        idx += 1

    return An

def ArcCentered(x,y,BlackBackground,jetter,Bound):
    orien = [i*np.pi/2 for i in range(4)] * 6
    s = [0.5]*4+[0.25]*4+[0.5]*4+[0.25]*4+[0.5]*4+[0.25]*4
    a = [np.pi/2]*8 + [np.pi]*8 + [3*np.pi/2]*8

    paras = zip(orien, s, a)
    if jetter:
        paras = pert.modulation(paras,'Arc')

    Ar = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        Ar[idx] = ig.ArcCentered(bounds=Bound,smoothing=0.015,thickness=0.05,arc_length=par[2],size=par[1],
                                 orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert Ar[idx].max() <= 1.0 and Ar[idx].min >= 0
            Ar[idx] = 1-Ar[idx]
        idx += 1

    return Ar

def TJunction_generator(x,y,BlackBackground,jetter,Bound):
    orien = [i*np.pi/4 for i in range(8)]*2
    s = [0.5]*8+[0.25]*8

    paras = zip(orien, s)
    if jetter:
        paras = pert.modulation(paras,'Bar')

    Tj = np.empty(len(paras),dtype=np.object)
    idx = 0
    for par in paras:
        Tj[idx] = TJunction(bounds=Bound,thickness=0.05,smoothing=0.015,size=par[1],
                            orientation=par[0], xdensity=x,ydensity=y)()
        if BlackBackground:
            assert Tj[idx].max() <= 1.0 and Tj[idx].min >= 0
            Tj[idx] = 1-Tj[idx]
        idx += 1
    return Tj



if __name__ == '__main__':
    # image size
    x = 64
    y = 64
    Bound = BoundingBox(radius=1)
    BlackBackground = True
    # convert image to black background
    jetter = True  # add perturbations

    B = Bar(x, y, BlackBackground, jetter, Bound)
    # I = Asterisk(x, y, BlackBackground, jetter, Bound)
    # C = Ring(x, y, BlackBackground, jetter, Bound)
    # L = Angle(x, y, BlackBackground, jetter, Bound)
    # A = ArcCentered(x, y, BlackBackground, jetter, Bound)
    # J = TJunction_generator(x, y, BlackBackground, jetter, Bound)
    # ############## save images to .mat file
    if jetter:
        if BlackBackground:
            sio.savemat('B_pert_b.mat', {'imgs': B})
            # sio.savemat('I_pert_b.mat', {'imgs': I})
            # sio.savemat('C_pert_b.mat', {'imgs': C})
            # sio.savemat('L_pert_b.mat', {'imgs': L})
            # sio.savemat('A_pert_b.mat', {'imgs': A})
            # sio.savemat('J_pert_b.mat', {'imgs': J})
        else:
            sio.savemat('B_pert_w.mat', {'imgs': B})
            # sio.savemat('I_pert_w.mat', {'imgs': I})
            # sio.savemat('C_pert_w.mat', {'imgs': C})
            # sio.savemat('L_pert_w.mat', {'imgs': L})
            # sio.savemat('A_pert_w.mat', {'imgs': A})
            # sio.savemat('J_pert_w.mat', {'imgs': J})
    else:
        if BlackBackground:
            sio.savemat('B_b.mat', {'imgs': B})
            # sio.savemat('I_b.mat', {'imgs': I})
            # sio.savemat('C_b.mat', {'imgs': C})
            # sio.savemat('L_b.mat', {'imgs': L})
            # sio.savemat('A_b.mat', {'imgs': A})
            # sio.savemat('J_b.mat', {'imgs': J})
        else:
            sio.savemat('B_w.mat', {'imgs': B})
            # sio.savemat('I_w.mat', {'imgs': I})
            # sio.savemat('C_w.mat', {'imgs': C})
            # sio.savemat('L_w.mat', {'imgs': L})
            # sio.savemat('A_w.mat', {'imgs': A})
            # sio.savemat('J_w.mat', {'imgs': J})
