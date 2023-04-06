import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io, color

#os.getcwd()
#os.chdir()


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

def plot_frame_1(t, X0, Y0, Z0, theta, traslacion, ax):
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

def create_animation_1(X0, Y0, Z0, theta, traslacion):
    N = len(theta) # = len(traslacion)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-45)
    ts = range(N) 
    ani = animation.FuncAnimation(fig, plot_frame_1, ts, fargs=[X0,Y0,Z0,theta, traslacion, ax], interval=10)
    plt.show()



def problema1(): 
    N = 100 # nº de frames
    X, Y, Z = axes3d.get_test_data(0.05) # test data
    X, Y = X - 50, Y - 50  # separar del centro (0,0) para apreciar la rotación
    d = Z.max()-Z.min()  # diámetro mayor
    theta = np.linspace(0, 3*np.pi, N)
    traslacion = np.linspace(0, d, N)
    create_animation_1(X, Y, Z, theta, traslacion)

##########################          PROBLEMA 2          ##########################

def get_xyz(img):

    # xx, yy -> coordenadas
    # zz -> imagen (verde)
    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    xx = np.asarray(xx).reshape(-1)
    yy = np.asarray(yy).reshape(-1)
    z  = img[:,:,1]
    zz = np.asarray(z).reshape(-1)

    # Consideraremos sólo los elementos con zz < 240
    x0 = xx[zz<240]
    y0 = yy[zz<240]
    z0 = zz[zz<240]/256. # -> normalizar los datos entre 0, 1.

    return x0, y0, z0

def transf1D(x,y,z,M, v=np.array([0,0,0])):
    xt = x*0
    yt = x*0
    zt = x*0
    for i in range(len(x)):
        q = np.array([x[i],y[i],z[i]])
        xt[i], yt[i], zt[i] = np.matmul(M, q) + v
    return xt, yt, zt

def plot_frame_2(t, X0, Y0, Z0, theta, traslacion, ax):
    
    # transformar los datos
    X, Y, Z = Rotacion(X0, Y0, Z0, theta[t])
    X, Y, Z = Traslacion(X, Y, Z, traslacion[t])

    # limpiar el gráfico
    ax.clear()
    ax.set_xlim(-400,400)
    ax.set_ylim(-400,400)

    # plot
    col = plt.get_cmap("viridis")(np.array(0.1+Z0))
    ax.scatter(X,Y,c=col,s=0.1)#,animated=True)

def create_animation_2(X0, Y0, Z0, theta, traslacion):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d', xlim=(0,400), ylim=(0,400))
    ax.view_init(elev=20, azim=-45)
    ax.set_title("Rotación de la imgen")
    ani = animation.FuncAnimation(fig, plot_frame_2, frames=np.arange(0,40), fargs=[X0, Y0, Z0, theta, traslacion, ax], interval=10)
    return ani

def problema2():
    img = io.imread('arbol.png')
    x0, y0, z0 = get_xyz(img)
    N = 50
    d = x0.max() - x0.min()
    theta = np.linspace(0, 3*np.pi, N)
    traslacion = np.linspace(0, d, N)
    ani = create_animation_2(x0, y0, z0, theta, traslacion)
    plt.show()
#os.chdir()
#ani.save("p4ii.gif", fps = 10)  
#os.getcwd()







if __name__ == "__main__":
    #problema1()
    problema2()