from SimulateCollisions import *

print("Initial values:")
print("m1: " +str(m1_Arr[0]) +"; v1: "+str(v1a[0,0])+"; px_1: "+str(sp.multiply(m1_Arr,v1a)[0,0]))
print("m2: " +str(m2_Arr[0]) +"; v2: "+str(v2a[0,0])+"; px_2: "+str(sp.multiply(m2_Arr,v2a)[0,0]))
print("Initial aligned momentum:: " + str(x_momentum(m1_Arr,m2_Arr,sp.transpose(v1a),sp.transpose(v2a))[0]))
print("")
print("Final values")
print("m1: " +str(m1_Arr[0]) +"; v1: "+str(v1a_total_f[0])+"; px_1: "+str(sp.multiply(m1_Arr,v1a_total_f)[0]))
print("m2: " +str(m2_Arr[0]) +"; v2: "+str(v2a_total_f[0])+"; px_2: "+str(sp.multiply(m2_Arr,v2a_total_f)[0]))
print("Final aligned momentum:: " + str(x_momentum(m1_Arr,m2_Arr,sp.transpose(v1a_f),sp.transpose(v2a_f))[0]))
print("")

print("-----------------------------------------Aligned Energy conserved: " + conservation(E_i, E_f) )
print("-----------------------------------------Aligned p_x conserved: " + conservation(p_x_i,p_x_f))
print("-----------------------------------------Aligned p_y conserved: " + conservation(p_y_i,p_y_f))

print("-----------------------------------------Lab Energy conserved: " + conservation(E_i, E_f) )
print("-----------------------------------------Lab p_x conserved: " + conservation(p_x_i,p_x_f))
print("-----------------------------------------Lab conserved: " + conservation(p_y_i,p_y_f))

# print("E_i = "+ str(E_i))
# print("p_x_i = "+ str(p_x_i))
# print("p_y_i = "+ str(p_y_i))
# print("E_f = "+ str(E_f))
# print("p_x_f = "+ str(p_x_f))
# print("p_y_f = "+ str(p_y_f))
