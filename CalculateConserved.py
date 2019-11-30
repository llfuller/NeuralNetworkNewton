import scipy as sp
# Methods for calculation of conserved quantities energy and momentum
def energy(m1, m2, v1, v2):
    v1_norm_squared = sp.array([sp.power(v1[trial][0],2) + sp.power(v1[trial][1],2) for trial in range(len(v1))])
    v2_norm_squared = sp.array([sp.power(v2[trial][0],2) + sp.power(v2[trial][1],2) for trial in range(len(v2))])
    E1 = sp.multiply(0.5,  sp.multiply(m1,v1_norm_squared))
    E2 = sp.multiply(0.5,  sp.multiply(m2,v2_norm_squared))
    return sp.real(E1 + E2)
def x_momentum(m1, m2, v1, v2):
    if (v1.shape)[-1] == 2:  # if second index of v1 is 2 (2 rows implies one for x, and one for y)
        v1_x = sp.array([v1[trial][0] for trial in range(len(v1))])
        v2_x = sp.array([v2[trial][0] for trial in range(len(v2))])
    else: # if given v1 is just v1_x
        v1_x = v1
        v2_x = v2
    p1 = sp.multiply(m1, v1_x)
    p2 = sp.multiply(m2, v2_x)
    return sp.real(p1 + p2)
def y_momentum(m1, m2, v1, v2):
    if (v1.shape)[-1] == 2:  # if second index of v1 is 2 (2 rows implies one for x, and one for y)
        v1_y = sp.array([v1[trial][1] for trial in range(len(v1))])
        v2_y = sp.array([v2[trial][1] for trial in range(len(v2))])
    else: # if given v1 is just v1_y
        v1_y = v1
        v2_y = v2
    p1 = sp.multiply(m1, v1_y)
    p2 = sp.multiply(m2, v2_y)
    return sp.real(p1 + p2)
def conservation(a_i, a_f):
    # print("a_i: " +str(a_i[0]))
    # print("a_f: " +str(a_f[0]))
    # print("error: " +str(sp.fabs(sp.divide((a_i - a_f),a_i))[0]))
    return str( False not in (sp.isclose(a_i,a_f)))