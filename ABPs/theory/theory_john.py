import sympy as sym
import numpy as np
phi = sym.Symbol('phi') #let this be the area fraction
phimax = sym.Symbol('phimax') #let this be the max area fraction
g = (1+0.2 * (phi/phimax)) * (1-(phi/phimax)) # let this be g(phi) at close contact

Ubar = 1/(1+ phi * g)
T = sym.Symbol('T')
k = sym.Symbol('k')
Uf = sym.Symbol('Uf')
U = sym.Symbol('U')
tauf = sym.Symbol('tauf')
zeta = sym.Symbol('z')
Us = sym.Symbol('Us')
taus = sym.Symbol('taus')
a = sym.Symbol('a')
ls = Us * taus
lf = Uf * tauf
tau = sym.Symbol('tau')
l = U * tau
chis = sym.Symbol('chis')
Ef = ( zeta * Uf * lf ) / (2 * k * T)
Es = ( zeta * Us * ls ) / (2 * k * T)
Emono = ( zeta * U * l ) / (2 * k * T)
active_press = phi * (1 + 2 * phi * g) + Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar + 4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * (chis * (Us/Uf) + (1-chis)) * g
active_press_John = phi * (1 + 2 * phi * g) + Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar + 4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * (chis * (Us/Uf) + (1-chis)) * g

g = (1+0.2 * (phi/phimax)) * (1-(phi/phimax)) # let this be g(phi) at close contact
Ubar = 1/(1+ phi * g)
active_press_mono = phi * (1+2*phi*g) + phi * phi * Emono * (a/l) * 4 * np.sqrt(2) * g + phi * Emono * Ubar

active_press_simp = sym.simplify(active_press)
active_press_simp_mono = sym.simplify(active_press_mono)
#print(active_press_simp)
#print(active_press_simp_mono)
#stop
#stop
first_derivative = sym.diff(active_press, phi)
#print(first_derivative)

second_derivative = sym.diff(first_derivative, phi)
second_derivative_simp = sym.simplify(second_derivative)
#print(second_derivative_simp)
#stop
first_derivative_mono = sym.diff(active_press_mono, phi)
#print(first_derivative_mono)
#stop
second_derivative_mono = sym.diff(first_derivative_mono, phi)
second_derivative_simp_mono = sym.simplify(second_derivative_mono)
#print(second_derivative_simp_mono)
#stop





phimax = 1.0#np.pi/(2 * np.sqrt(3))
z = 1.0
k = 1.0
T = 1.0
zeta = 1.0

Us = np.linspace(0, 500, 20)#input
Uf = np.linspace(0, 500, 20)#input
chis = 0.5#np.ones(50) * 0.5
a = 0.5 #calculate this
taus = 1/3
tauf = 1/3
tau = 1/3
ls = Us * taus
lf = Uf * tauf
l = U * tau
#SOLVE FOR PHI
phi = sym.Symbol('phi', positive=True, real=True) #let this be the area fraction
import matplotlib.pyplot as plt


first_derivate = np.zeros((len(Us), len(Uf)))

for i in range(0, len(Us)):
    print(i)
    for j in range(0, len(Uf)):
        test_result = sym.solve(2*phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + phi*(0.4*phi*(-phi/phimax + 1)/phimax - 2*phi*(0.2*phi/phimax + 1)/phimax + (-2*phi/phimax + 2)*(0.2*phi/phimax + 1)) + 1 + Uf[j]**2*phi*tauf*z*(chis*(-1 + Us[i]**2*taus/(Uf[j]**2*tauf)) + 1)*(-0.2*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/phimax - (-phi/phimax + 1)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)**2) + Uf[j]**2*tauf*z*(chis*(-1 + Us[i]**2*taus/(Uf[j]**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)) + 0.565685424949238*Uf[j]*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us[i]*chis/Uf[j])/(T*k*phimax) - 2.82842712474619*Uf[j]*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us[i]*chis/Uf[j])/(T*k*phimax) + 5.65685424949238*Uf[j]*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us[i]*chis/Uf[j])/(T*k), phi)
        print(test_result)
        print(len(test_result))
        
        if len(test_result)>0:
            first_derivate[i,j] = test_result[0]
        else:
            first_derivate[i,j] = 0

plt.contourf(Us, Uf, first_derivate)
plt.colorbar()
plt.show()
stop







phimax = 0.64#np.pi/(2 * np.sqrt(3))
z = 1.0
k = 1.0
T = 1.0
zeta = 1.0

Us = 150#500#np.linspace(0, 500, 50)#input
Uf = 500#np.linspace(0, 500, 50)#input
U = 500#np.linspace(0, 500, 50)#input
chis = np.linspace(0, 1.0, 50) #input
chis = 0.5#np.ones(50) * 0.5
a = 0.5 #calculate this
taus = 1/3
tauf = 1/3
tau = 1/3
ls = Us * taus
lf = Uf * tauf
l = U * tau
#SOLVE FOR PHI
phi = sym.Symbol('phi', positive=True, real=True) #let this be the area fraction
import matplotlib.pyplot as plt
phi = np.linspace(0, 0.8, 50)

print(phi)
print(Us)
print(Uf)
active_press = phi*(-2*T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) - 5.65685424949238*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) + phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))
print(-2*T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))
print(- 5.65685424949238*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))
print(phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))

g = (1+0.2 * (phi/phimax)) * (1-(phi/phimax)) # let this be g(phi) at close contact
Ubar = 1/(1+ phi * g)
ls = Us * taus
lf = Uf * tauf
Ef = ( zeta * Uf * lf ) / (2 * k * T)
Es = ( zeta * Us * ls ) / (2 * k * T)
active_press_new = phi * (1 + 2 * phi * g) + Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar + 4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * (chis * (Us/Uf) + (1-chis)) * g
print(phi * (1 + 2 * phi * g))
print(Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar)
print(4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * (chis * (Us/Uf) + (1-chis)) * g)

#active_press_4 = phi*(-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)*(phi*(0.2*phi + phimax)*(phi - phimax) - 2*phimax**2) - 1.4142135623731*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2) + 4*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2))
active_press_mono = -phi*(2*T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) + U**2*phimax**4*tau*z + 5.65685424949238*U*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))

first_derivate = 2*phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + phi*(0.4*phi*(-phi/phimax + 1)/phimax - 2*phi*(0.2*phi/phimax + 1)/phimax + (-2*phi/phimax + 2)*(0.2*phi/phimax + 1)) + 1 + Uf**2*phi*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)*(-0.2*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/phimax - (-phi/phimax + 1)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)**2) + Uf**2*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)) + 0.565685424949238*Uf*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) - 2.82842712474619*Uf*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) + 5.65685424949238*Uf*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k)
#first_derivate_4 = phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/2 + phi*(0.1*phi*(-phi/phimax + 1)/phimax - phi*(0.2*phi/phimax + 1)/(2*phimax) + (-phi/(2*phimax) + 1/2)*(0.2*phi/phimax + 1)) + 1 + Uf**2*phi*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)*(-0.05*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/(4*phimax) - (-phi/(4*phimax) + 1/4)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/4 + 1)**2) + Uf**2*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/4 + 1)) + 0.14142135623731*Uf*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) - 0.707106781186548*Uf*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) + 1.4142135623731*Uf*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k)
second_derivate = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(8*phi*(0.2*phi + phimax) + 1.6*phi*(phi - phimax) + 2*phi*(2.4*phi + 3.2*phimax) + 8*(0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(2.26274169979695*phi**2 + 22.6274169979695*phi*(0.2*phi + phimax) + 4.5254833995939*phi*(phi - phimax) + 11.3137084989848*(0.2*phi + phimax)*(phi - phimax)) + phi*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) - phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + phi*(1.2*phi + 1.6*phimax) + 2*(0.2*phi + phimax)*(phi - phimax)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3)
#second_derivate_4 = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)**3*(phi*(0.2*phi + phimax) + phi*(0.6*phi + 0.8*phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)**3*(0.282842712474619*phi**2 + 2.82842712474619*phi*(0.2*phi + phimax) + 0.565685424949238*phi*(phi - phimax) + 1.4142135623731*(0.2*phi + phimax)*(phi - phimax)) + 4*phi*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))**2 - 4*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)*(phi*(0.2*phi + phimax) + phi*(0.6*phi + 0.8*phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax)))/(T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)**3)

first_derivate_mono = 2*phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + phi*(0.4*phi*(-phi/phimax + 1)/phimax - 2*phi*(0.2*phi/phimax + 1)/phimax + (-2*phi/phimax + 2)*(0.2*phi/phimax + 1)) + 1 + U**2*phi*tau*z*(-0.2*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/phimax - (-phi/phimax + 1)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)**2) + U**2*tau*z/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)) + 0.565685424949238*U*a*phi**2*z*(-phi/phimax + 1)/(T*k*phimax) - 2.82842712474619*U*a*phi**2*z*(0.2*phi/phimax + 1)/(T*k*phimax) + 5.65685424949238*U*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/(T*k)
second_derivate_mono = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(8*phi*(0.2*phi + phimax) + 1.6*phi*(phi - phimax) + 2*phi*(2.4*phi + 3.2*phimax) + 8*(0.2*phi + phimax)*(phi - phimax)) - U**2*phi*phimax**4*tau*z*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) + U**2*phimax**4*tau*z*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + phi*(1.2*phi + 1.6*phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) - U*a*z*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(2.26274169979695*phi**2 + 22.6274169979695*phi*(0.2*phi + phimax) + 4.5254833995939*phi*(phi - phimax) + 11.3137084989848*(0.2*phi + phimax)*(phi - phimax)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3)




Us = 0#500#np.linspace(0, 500, 50)#input
Uf = 500#np.linspace(0, 500, 50)#input
ls = Us * taus
lf = Uf * tauf

active_press2 = phi*(-2*T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) - 5.65685424949238*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) + phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))
first_derivate2 = 2*phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + phi*(0.4*phi*(-phi/phimax + 1)/phimax - 2*phi*(0.2*phi/phimax + 1)/phimax + (-2*phi/phimax + 2)*(0.2*phi/phimax + 1)) + 1 + Uf**2*phi*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)*(-0.2*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/phimax - (-phi/phimax + 1)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)**2) + Uf**2*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)) + 0.565685424949238*Uf*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) - 2.82842712474619*Uf*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) + 5.65685424949238*Uf*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k)
second_derivate2 = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(8*phi*(0.2*phi + phimax) + 1.6*phi*(phi - phimax) + 2*phi*(2.4*phi + 3.2*phimax) + 8*(0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(2.26274169979695*phi**2 + 22.6274169979695*phi*(0.2*phi + phimax) + 4.5254833995939*phi*(phi - phimax) + 11.3137084989848*(0.2*phi + phimax)*(phi - phimax)) + phi*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) - phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + phi*(1.2*phi + 1.6*phimax) + 2*(0.2*phi + phimax)*(phi - phimax)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3)

plt.scatter(phi, active_press)

#plt.scatter(phi, active_press2)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\Pi$')

#plt.scatter(phi, active_press_4)
plt.tight_layout()
plt.show()

plt.scatter(phi, first_derivate)
#plt.scatter(phi, first_derivate2)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\frac{\partial \Pi}{\partial \phi}$')
#plt.scatter(phi, first_derivate_4)
plt.tight_layout()
plt.show()

plt.scatter(phi, second_derivate)
#plt.scatter(phi, second_derivate2)
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\frac{\partial^2 \Pi}{\partial \phi^2}$')
plt.tight_layout()
#plt.scatter(phi, second_derivate_4)
plt.show()
"""
stop
second_derivative = np.zeros((len(Us), len(Uf)))
for i in range(0, len(Us)):
    for j in range(0, len(Uf)):
        #for k in range(0, len(chis)):
        print(sym.solve((-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(8*phi*(0.2*phi + phimax) + 1.6*phi*(phi - phimax) + 2*phi*(2.4*phi + 3.2*phimax) + 8*(0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf[j]*(1 - chis) + Us[i]*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(2.26274169979695*phi**2 + 22.6274169979695*phi*(0.2*phi + phimax) + 4.5254833995939*phi*(phi - phimax) + 11.3137084989848*(0.2*phi + phimax)*(phi - phimax)) + phi*phimax**4*z*(-Uf[j]**2*tauf + chis*(Uf[j]**2*tauf - Us[i]**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) - phimax**4*z*(-Uf[j]**2*tauf + chis*(Uf[j]**2*tauf - Us[i]**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + phi*(1.2*phi + 1.6*phimax) + 2*(0.2*phi + phimax)*(phi - phimax)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3)))


print(second_derivative)

"""






Us = np.linspace(0, 500, 50)#input
Uf = 500#np.linspace(0, 500, 50)#input
chis = 0.5#np.ones(50) * 0.5
a = 0.5 #calculate this
taus = 1/3
tauf = 1/3
tau = 1/3
ls = Us * taus
lf = Uf * tauf
l = U * tau
#SOLVE FOR PHI
phi = 0.6

active_press = phi*(-2*T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) - 5.65685424949238*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) + phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))

g = (1+0.2 * (phi/phimax)) * (1-(phi/phimax)) # let this be g(phi) at close contact
Ubar = 1/(1+ phi * g)
ls = Us * taus
lf = Uf * tauf
Ef = ( zeta * Uf * lf ) / (2 * k * T)
Es = ( zeta * Us * ls ) / (2 * k * T)
active_press_new = phi * (1 + 2 * phi * g) + Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar + 4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * (chis * (Us/Uf) + (1-chis)) * g
print(phi * (1 + 2 * phi * g))
print(Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar)
print(Ubar)
print(4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * (chis * (Us/Uf) + (1-chis)) * g)

#stop
#active_press_4 = phi*(-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)*(phi*(0.2*phi + phimax)*(phi - phimax) - 2*phimax**2) - 1.4142135623731*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2) + 4*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2))

first_derivate = 2*phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + phi*(0.4*phi*(-phi/phimax + 1)/phimax - 2*phi*(0.2*phi/phimax + 1)/phimax + (-2*phi/phimax + 2)*(0.2*phi/phimax + 1)) + 1 + Uf**2*phi*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)*(-0.2*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/phimax - (-phi/phimax + 1)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)**2) + Uf**2*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)) + 0.565685424949238*Uf*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) - 2.82842712474619*Uf*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) + 5.65685424949238*Uf*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k)
#first_derivate_4 = phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/2 + phi*(0.1*phi*(-phi/phimax + 1)/phimax - phi*(0.2*phi/phimax + 1)/(2*phimax) + (-phi/(2*phimax) + 1/2)*(0.2*phi/phimax + 1)) + 1 + Uf**2*phi*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)*(-0.05*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/(4*phimax) - (-phi/(4*phimax) + 1/4)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/4 + 1)**2) + Uf**2*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1)/4 + 1)) + 0.14142135623731*Uf*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) - 0.707106781186548*Uf*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) + 1.4142135623731*Uf*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k)
second_derivate = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(8*phi*(0.2*phi + phimax) + 1.6*phi*(phi - phimax) + 2*phi*(2.4*phi + 3.2*phimax) + 8*(0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(2.26274169979695*phi**2 + 22.6274169979695*phi*(0.2*phi + phimax) + 4.5254833995939*phi*(phi - phimax) + 11.3137084989848*(0.2*phi + phimax)*(phi - phimax)) + phi*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) - phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + phi*(1.2*phi + 1.6*phimax) + 2*(0.2*phi + phimax)*(phi - phimax)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3)
#second_derivate_4 = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)**3*(phi*(0.2*phi + phimax) + phi*(0.6*phi + 0.8*phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)**3*(0.282842712474619*phi**2 + 2.82842712474619*phi*(0.2*phi + phimax) + 0.565685424949238*phi*(phi - phimax) + 1.4142135623731*(0.2*phi + phimax)*(phi - phimax)) + 4*phi*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))**2 - 4*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)*(phi*(0.2*phi + phimax) + phi*(0.6*phi + 0.8*phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax)))/(T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - 4*phimax**2)**3)



Us = np.linspace(0, 500, 50)#input
Uf = 150#np.linspace(0, 500, 50)#input
ls = Us * taus
lf = Uf * tauf

active_press2 = phi*(-2*T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) - 5.65685424949238*a*phi*z*(0.2*phi + phimax)*(phi - phimax)*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2) + phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2))
first_derivate2 = 2*phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + phi*(0.4*phi*(-phi/phimax + 1)/phimax - 2*phi*(0.2*phi/phimax + 1)/phimax + (-2*phi/phimax + 2)*(0.2*phi/phimax + 1)) + 1 + Uf**2*phi*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)*(-0.2*phi*(-phi/phimax + 1)/phimax + phi*(0.2*phi/phimax + 1)/phimax - (-phi/phimax + 1)*(0.2*phi/phimax + 1))/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)**2) + Uf**2*tauf*z*(chis*(-1 + Us**2*taus/(Uf**2*tauf)) + 1)/(2*T*k*(phi*(-phi/phimax + 1)*(0.2*phi/phimax + 1) + 1)) + 0.565685424949238*Uf*a*phi**2*z*(-phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) - 2.82842712474619*Uf*a*phi**2*z*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k*phimax) + 5.65685424949238*Uf*a*phi*z*(-phi/phimax + 1)*(0.2*phi/phimax + 1)*(-chis + 1 + Us*chis/Uf)/(T*k)
second_derivate2 = (-T*k*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(8*phi*(0.2*phi + phimax) + 1.6*phi*(phi - phimax) + 2*phi*(2.4*phi + 3.2*phimax) + 8*(0.2*phi + phimax)*(phi - phimax)) - a*z*(Uf*(1 - chis) + Us*chis)*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3*(2.26274169979695*phi**2 + 22.6274169979695*phi*(0.2*phi + phimax) + 4.5254833995939*phi*(phi - phimax) + 11.3137084989848*(0.2*phi + phimax)*(phi - phimax)) + phi*phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax) + 0.2*phi*(phi - phimax) + (0.2*phi + phimax)*(phi - phimax))*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + 2*(0.2*phi + phimax)*(phi - phimax)) - phimax**4*z*(-Uf**2*tauf + chis*(Uf**2*tauf - Us**2*taus))*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)*(2*phi*(0.2*phi + phimax) + 0.4*phi*(phi - phimax) + phi*(1.2*phi + 1.6*phimax) + 2*(0.2*phi + phimax)*(phi - phimax)))/(2*T*k*phimax**2*(phi*(0.2*phi + phimax)*(phi - phimax) - phimax**2)**3)

active_press_john = phi * (1 + 2 * phi * g) + Ef * phi * (1 + chis * (Es/Ef-1)) * Ubar + 4 * np.sqrt(2) * phi * phi *Ef * (a/lf) * ((1+chis)*(1-chis) + chis*(2-chis) * (Us/Uf)) * g


#plt.scatter(Us, active_press)

plt.scatter(Us, active_press_new)
#plt.scatter(Us, active_press_john)

plt.xlabel(r'$\mathrm{Pe}_\mathrm{S}$')
plt.ylabel(r'$\Pi$')

#plt.scatter(phi, active_press_4)
plt.tight_layout()
plt.show()

plt.scatter(Us, first_derivate)
#plt.scatter(Us, first_derivate2)
plt.xlabel(r'$\mathrm{Pe}_\mathrm{S}$')
plt.ylabel(r'$\frac{\partial \Pi}{\partial \phi}$')
#plt.scatter(phi, first_derivate_4)
plt.tight_layout()
plt.show()

plt.scatter(Us, second_derivate)
#plt.scatter(Us, second_derivate2)
plt.xlabel(r'$\mathrm{Pe}_\mathrm{S}$')
plt.ylabel(r'$\frac{\partial^2 \Pi}{\partial \phi^2}$')
plt.tight_layout()
#plt.scatter(phi, second_derivate_4)
plt.show()