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
plt.title("Collision")
# plt.plot(Position1L_t_pre[0,0,:], Position1L_t_pre[1,0,:], label="1")
# plt.plot(Position2L_t_pre[0,0,:], Position2L_t_pre[1,0,:], label="2")
#
# plt.plot(Position1L_t_post[0,0,:], Position1L_t_post[1,0,:], label="1")
# plt.plot(Position2L_t_post[0,0,:], Position2L_t_post[1,0,:], label="2")
plt.legend()

plt.figure()
print(sp.shape(t))
print(sp.shape(Velocity1L_t))
plt.plot(t[:,0], Velocity1L_t[0,0,:], label="Particle 1")
plt.plot(t[:,0], Velocity2L_t[0,0,:], label="Particle 2")
plt.title("V_x")
plt.legend()
plt.show()
