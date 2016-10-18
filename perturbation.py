import numpy as np

def modulation_of_jetter(para_list,shape_name):
    para1_modulation = [-12,-9,-6,-3,0,3,6,9,12]
    para2_modulation = [0.826,0.909,1,1.1,1.21]

    p1_m_len = len(para1_modulation)
    p2_m_len = len(para2_modulation)
    num_pert = p1_m_len * p2_m_len
    new_para_list = [None] * num_pert * len(para_list)
    if shape_name in ['SG']:
        for i in range(len(para_list)):
            new_para_list[num_pert*i:num_pert*(i+1)] = [(para_list[i][0] + np.pi/180*j, para_list[i][1] * k )
                                                        for j in para1_modulation for k in para2_modulation]
    elif shape_name == 'HG':
        for i in range(len(para_list)):
            new_para_list[num_pert*i:num_pert*(i+1)] = [(para_list[i][0] + np.pi/180*j, para_list[i][1] * k,
                                                         para_list[i][2] * k, para_list[i][3]*k)
                                                        for j in para1_modulation for k in para2_modulation]
    elif shape_name == 'SpG':
        for i in range(len(para_list)):
            new_para_list[num_pert*i:num_pert*(i+1)] = [(para_list[i][0] + np.pi/180*j, para_list[i][1] * k,
                                                         para_list[i][2] * k, para_list[i][3])
                                                        for j in para1_modulation for k in para2_modulation]
    elif shape_name in ['RG', 'Asterisk', 'Angle', 'Arc', 'Bar']:
        for i in range(len(para_list)):
            new_para_list[num_pert*i:num_pert*(i+1)] = [(para_list[i][0] + np.pi/180*j, para_list[i][1] * k,
                                                        para_list[i][2])
                                                        for j in para1_modulation for k in para2_modulation]
    elif shape_name in ['T', 'Ring']:
        ratio = [1/1.1]*8 + [1]
        for i in range(len(para_list)):
            new_para_list[num_pert*i:num_pert*(i+1)] = [(j*np.pi/8, para_list[i][1] * k,para_list[i][2] * k,
                                                         para_list[i][3] * k, ratio[j])
                                                        for j in range(9) for k in para2_modulation]
    return new_para_list




def modulation_of_phase(para_list,shape_name,frames):
    new_para_list = np.empty([len(para_list),frames],dtype=list)
    if shape_name in ['SG','HG']:
        for i in range(len(para_list)):
            new_para_list[i][0:frames] = [(para_list[i][0][0], para_list[i][0][1], para_list[i][0][2]+ 2*np.pi*j/frames)
                                                        for j in range(frames)]
    elif shape_name in ['SpG']:
        for i in range(len(para_list)):
            new_para_list[i][0:frames]  = [(para_list[i][0][0] + 2*np.pi*j/frames/para_list[i][0][2], para_list[i][0][1],
                                                         para_list[i][0][2]) for j in range(frames)]
    elif shape_name in ['RG']:
        for i in range(len(para_list)):
            new_para_list[i][0:frames]  = [(para_list[i][0][0] + 2*np.pi*j/frames/para_list[i][0][1], para_list[i][0][1])
                                          for j in range(frames)]
    elif shape_name in ['T']:
        for i in range(len(para_list)):
            new_para_list[i][0:frames]  = [(para_list[i][0][0], para_list[i][0][1], para_list[i][0][2],
                                                         para_list[i][0][3]+2*np.pi*j/frames) for j in range(frames)]
    return new_para_list