from SimulateCollisions import *

#Same:
# print("Particle 1 t=0 x_a: " + str(x1a_Arr[0]))
# print("Particle 2 t=0 x_a: " + str(x2a_Arr[0]))
# print("InitialPosition1a: " + str(InitialPosition1a[0][0]))
# print("InitialPosition2a: " + str(InitialPosition2a[0][0]))
# print("InitialPosition1n length: "+str(sp.sqrt(sp.power(InitialPosition1n[0][0],2)+sp.power(InitialPosition1n[0][1],2))))
# print("v_cm: "+str(vcm))

print("Particle 1 time to reach x=0: " + str(sp.fabs(sp.divide(x1a_Arr[0],v1ax_Arr[0]))))
print("Particle 2 time to reach x=0: " + str(sp.fabs(sp.divide(x2a_Arr[0],v2ax_Arr[0]))))
print("Particle 1 initial position: " + str(Position1L_t[:,0,0]) + " at v_x="+str(v1L[0][0]))
print("Particle 2 initial position: " + str(Position2L_t[:,0,0]) + " at v_x="+str(v2L[0][0]))
print("Angle of rotation: " + str(sp.divide(phi[0],sp.pi)) +" PI radians")

plt.figure()
plt.plot(Position1L_t[0,0,:], Position1L_t[1,0,:], label="Particle 1")
plt.plot(Position2L_t[0,0,:], Position2L_t[1,0,:], label="Particle 2")
plt.title("Collision:"+ str(round(sp.divide(phi[0],sp.pi),2)) +" PI radians and v_cm="+"("+str(round(vcm_x_Arr[0],3))+","
          +str(round(vcm_y_Arr[0],3))+")" + " and m="+"("+str(round(m1_Arr[0],2))+","+str(round(m2_Arr[0],2))+")")
# plt.plot(Position1L_t_pre[0,0,:], Position1L_t_pre[1,0,:], label="1")
# plt.plot(Position2L_t_pre[0,0,:], Position2L_t_pre[1,0,:], label="2")
#
# plt.plot(Position1L_t_post[0,0,:], Position1L_t_post[1,0,:], label="1")
# plt.plot(Position2L_t_post[0,0,:], Position2L_t_post[1,0,:], label="2")
plt.legend()
print(sp.shape(Velocity1L_t))
plt.figure()
plt.plot(t[:,0], Velocity1L_t[0,0,:], label="Particle 1") # dimension of V: (2, numTrials, numTimeSteps)
plt.plot(t[:,0], Velocity2L_t[1,0,:], label="Particle 2")
plt.title("V_x")
plt.xlabel("Time")
plt.ylabel("v_x")
plt.legend()

plt.figure()
plt.plot(t[:,0], Velocity1L_t[0,0,:], label="Particle 1")
plt.plot(t[:,0], Velocity2L_t[0,0,:], label="Particle 2")
plt.title("V_y")
plt.xlabel("Time")
plt.ylabel("v_y")
plt.legend()

print("Particle 1 at t=0:")
print("Timestep size: "+str(round(dt[0],2)))
print("Position1: ")
print(Position1L_t[:,0,:])
print("Velocity1: ")
print(Velocity1L_t[:,0,:])
print("Particle 2 at t=0:")
print("Position2: ")
print(Position2L_t[:,0,:])
print("Velocity2: ")
print(Velocity2L_t[:,0,:])


plt.show()
