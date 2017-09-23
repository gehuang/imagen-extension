# generate 32 sinewaves for mapping RF tuning: 16 orientations and 2 frequencies

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

    orientations = [i * np.pi /16 for i in range(16)]
    frequencies = [6.0, 12.0]
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



if __name__ == '__main__':
    # image size
    x = 200
    y = 200
    Bound = BoundingBox(radius=2)
    BlackBackground = True  # convert image to black background
    phase_shift = False  # generate different phases
    f = 4

    ############ waves
    SG = SineG(x, y, BlackBackground, phase_shift, Bound, frames=f)
    sio.savemat('SG.mat', {'imgs': SG})
