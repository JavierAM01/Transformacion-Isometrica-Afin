import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color

#os.getcwd()
#os.chdir()


"""
Ejemplo para el apartado 1.

Modifica la figura 3D y/o cambia el color
https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
"""


"""
Transformación para el segundo apartado

NOTA: Para el primer aparado es necesario adaptar la función o crear otra similar
pero teniendo en cuenta más dimensiones
"""

# Matriz de rotación
M = lambda theta : np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
M2 = lambda theta : np.array([[[np.cos(theta)]*3, [-np.sin(theta)]*3, [0]*3], [[np.sin(theta)]*3, [np.cos(theta)]*3, [0]*3], [[0]*3, [0]*3, [1]*3]])

def Rotacion_M(X0, Y0, Z0, M): 
    X = np.dot(M[0,0], X0) + np.dot(M[0,1], Y0) + np.dot(M[0,2], Z0)
    Y = np.dot(M[1,0], X0) + np.dot(M[1,1], Y0) + np.dot(M[1,2], Z0)
    Z = np.dot(M[2,0], X0) + np.dot(M[2,1], Y0) + np.dot(M[2,2], Z0)
    return X, Y, Z

def Rotacion(X0, Y0, Z0, theta): 
    X = np.dot(np.cos(theta), X0) + np.dot(-np.sin(theta), Y0)
    Y = np.dot(np.sin(theta), X0) + np.dot(np.cos(theta), Y0)
    Z = Z0
    return X, Y, Z

def Traslacion(X0, Y0, Z0, t):
    a, b, c = 0, 0, 1
    X, Y, Z = a*t+X0, b*t + Y0, c*t+Z0
    return X, Y, Z

def plot_frame(t, X0, Y0, Z0, theta, traslacion, ax):
    # matrix = M(theta[t])
    # X, Y, Z = Rotacion_M(X0, Y0, Z0, matrix)
    X, Y, Z = Rotacion(X0, Y0, Z0, theta[t])
    X, Y, Z = Traslacion(X, Y, Z, traslacion[t])
    ax.clear()
    ax.set_xlim(-75, 75)
    ax.set_ylim(-75, 75)
    ax.set_zlim(-100, 250)
    cset = ax.contour(X, Y, Z, 16, extend3d=True, cmap=plt.cm.get_cmap('viridis'))
    ax.clabel(cset, fontsize=9, inline=1)

def create_animation(X0, Y0, Z0, theta, traslacion):
    N = len(theta) # = len(traslacion)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-45)
    ts = range(N) 
    ani = animation.FuncAnimation(fig, plot_frame, ts, fargs=[X0,Y0,Z0,theta, traslacion, ax], interval=10)
    plt.show()



def problema1(): 
    N = 100 # nº de frames
    X, Y, Z = axes3d.get_test_data(0.05) # test data
    X, Y = X - 50, Y - 50  # separar del centro (0,0) para apreciar la rotación
    d = Z.max()-Z.min()  # diámetro mayor
    theta = np.linspace(0, 3*np.pi, N)
    traslacion = np.linspace(0, d, N)
    create_animation(X, Y, Z, theta, traslacion)



def transf1D(x,y,z,M, v=np.array([0,0,0])):
    xt = x*0
    yt = x*0
    zt = x*0
    for i in range(len(x)):
        q = np.array([x[i],y[i],z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
    return xt, yt, zt



"""
Segundo apartado casi regalado

Imagen del árbol
"""

#os.getcwd()
#os.chdir()

img = io.imread('arbol.png')
#dimensions = color.guess_spatial_dimensions(img)
#print(dimensions)
#io.show()
#io.imsave('arbol2.png',img)

#https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
fig = plt.figure(figsize=(5,5))
for i in range(3):
    ax = fig.add_subplot(1, 3, i+1)
    ax.set_title(f"img[:,:,{i}]")
    ax.contourf(img[:,:,i], cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
    ax.axis('off')
#fig.colorbar(p)

xyz = img.shape

x = np.arange(0,xyz[0],1)
y = np.arange(0,xyz[1],1)
xx,yy = np.meshgrid(x, y)
xx = np.asarray(xx).reshape(-1)
yy = np.asarray(yy).reshape(-1)
z  = img[:,:,1]
zz = np.asarray(z).reshape(-1)


"""
Consideraremos sólo los elementos con zz < 240 

Por curiosidad, comparamos el resultado con contourf y scatter!
"""
#Variables de estado coordenadas
x0 = xx[zz<240]
y0 = yy[zz<240]
z0 = zz[zz<240]/256.
#Variable de estado: color
col = plt.get_cmap("viridis")(np.array(0.1+z0))

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 2, 1)
ax.set_title("x, y, z")
plt.contourf(x,y,z,cmap = plt.cm.get_cmap('viridis'), levels=np.arange(0,240,2))
ax = fig.add_subplot(1, 2, 2)
ax.set_title("x0, y0")
plt.scatter(x0,y0,c=col,s=0.1)
plt.show()



def animate(t, ax):
    M = np.array([[1,0,0],[0,1,0],[0,0,1]])
    v=np.array([40,40,0])*t
    
    ax.clear()
    ax.set_xlim(0,400)
    ax.set_ylim(0,400)

    X, Y, Z = transf1D(x0, y0, z0, M=M, v=v)
    col = plt.get_cmap("viridis")(np.array(0.1+z0))
    ax.scatter(X,Y,c=col,s=0.1)#,animated=True)


fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(121, projection='3d', xlim=(0,400), ylim=(0,400))
ax.view_init(elev=20, azim=-45)
ax.set_title("Animation transf1D")
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,1,0.025), fargs=[ax], interval=20)
#os.chdir()
#ani.save("p4ii.gif", fps = 10)  
#os.getcwd()







if __name__ == "__main__":
    problema1()