import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import axes3d
from matplotlib import animation
from skimage import io

#os.getcwd()
#os.chdir()


# Crear un matriz de rotación, en el plano XY, de ángulo theta
M = lambda theta : np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

# rotar los valores : X0, Y0, Z0, con la matriz de rotación M
def Rotacion_M(X0, Y0, Z0, M): 
    X = np.dot(M[0,0], X0) + np.dot(M[0,1], Y0) + np.dot(M[0,2], Z0)
    Y = np.dot(M[1,0], X0) + np.dot(M[1,1], Y0) + np.dot(M[1,2], Z0)
    Z = np.dot(M[2,0], X0) + np.dot(M[2,1], Y0) + np.dot(M[2,2], Z0)
    return X, Y, Z

# rotar los valores : X0, Y0, Z0, en el plano XY, un ángulo theta.
def Rotacion(X0, Y0, Z0, theta): 
    xc, yc = X0.mean(), Y0.mean()
    X = np.dot(np.cos(theta), X0-xc) + np.dot(-np.sin(theta), Y0-yc) + xc
    Y = np.dot(np.sin(theta), X0-xc) + np.dot(np.cos(theta), Y0-yc) + yc
    Z = Z0
    return X, Y, Z

# trasladar X0, Y0, Z0
def Traslacion(X0, Y0, Z0, t, a=0, b=0, c=0):
    X, Y, Z = a*t+X0, b*t + Y0, c*t+Z0
    return X, Y, Z

# crear la imagen de un frame (del gif) : rotamos los datos y los ploteamos
def plot_frame_1(t, X0, Y0, Z0, theta, traslacion, ax):

    # transformar datos
    X, Y, Z = Rotacion(X0, Y0, Z0, theta[t])
    X, Y, Z = Traslacion(X, Y, Z, traslacion[t], c=1)

    # settings
    ax.clear()
    ax.set_xlim(-100, 0)
    ax.set_ylim(-100, 0)
    ax.set_zlim(-100, 250)
    ax.set_title("Rotación + Traslación : figura 3D")
    
    # plot
    ax.scatter(X.mean(), Y.mean(), -100, c="black")
    ax.contour(X, Y, Z, 16, extend3d=True, cmap=plt.cm.get_cmap('viridis'), zorder=1)
    # ax.clabel(cset, fontsize=9, inline=1)

# crear animación del ejercicio 1
def create_animation_1(X0, Y0, Z0, theta, traslacion):
    N = len(theta) # = len(traslacion)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=20, azim=-45)
    ts = range(N) 
    ani = animation.FuncAnimation(fig, plot_frame_1, ts, fargs=[X0,Y0,Z0,theta, traslacion, ax], interval=10)
    return ani


# extraer X, Y, Z de la imagen, para poder graficarla
# xx, yy -> coordenadas
# zz -> imagen (verde) -> con estos valores podremos indicar el color "zt" en el punto "(xt,yt)"
def get_xyz(img, K=240):

    x = np.arange(0, img.shape[0], 1)
    y = np.arange(0, img.shape[1], 1)
    xx, yy = np.meshgrid(x, y)
    xx = np.asarray(xx).reshape(-1)
    yy = np.asarray(yy).reshape(-1)
    z  = img[:,:,1]
    zz = np.asarray(z).reshape(-1)

    # Considerar sólo los elementos con zz < 240
    x0 = xx[zz<K]
    y0 = yy[zz<K]
    z0 = zz[zz<K]/256. # -> normalizar los datos entre 0, 1.

    return x0, y0, z0

# crear la imagen de un frame (del gif) : rotamos los datos y los ploteamos
def plot_frame_2(t, X0, Y0, Z0, theta, traslacion, ax):
    
    # transformar los datos
    X, Y, Z = Rotacion(X0, Y0, Z0, theta[t])
    X, Y, Z = Traslacion(X, Y, Z, traslacion[t], a=1, b=1) 
    xc, yc = X.mean(), Y.mean()

    # settings
    ax.clear()
    ax.set_xlim(-400,300)
    ax.set_ylim(400,1100)
    ax.set_zlim(-1,1)
    ax.set_title("Rotación + Traslación : imagen 2D")

    # plot
    col = plt.get_cmap("viridis")(np.array(0.1+Z0))
    image = ax.scatter(X, Y, 1, c=col, s=0.1) #,animated=True)
    ax.scatter(xc, yc, 0, c="black")

    return [image]

# crear animación del ejercicio 2
def create_animation_2(X0, Y0, Z0, theta, traslacion):
    N = len(theta)
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-45)
    ani = animation.FuncAnimation(fig, plot_frame_2, frames=range(0,N), fargs=[X0, Y0, Z0, theta, traslacion, ax], interval=10)
    return ani


# Resultados 

def problema1(save_gif=False): 
    N = 100 # nº de frames
    X, Y, Z = axes3d.get_test_data(0.05) # test data
    X, Y = X - 50, Y - 50  # separar del centro (0,0) para apreciar la rotación
    d = Z.max()-Z.min()  # diámetro mayor
    theta = np.linspace(0, 3*np.pi, N)
    traslacion = np.linspace(0, d, N)
    ani = create_animation_1(X, Y, Z, theta, traslacion)
    if save_gif:
        ani.save("images/figura_3d.gif", fps = 10) 
    plt.show()

def problema2(save_gif=False):

    img = io.imread('images/arbol.png')
    x0, y0, z0 = get_xyz(img)
    xc, yc = x0.mean(), y0.mean() 

    # plot : imagen + centroide
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.scatter(x0, y0, c=z0, s=0.1, cmap=plt.get_cmap("viridis"), vmin=0, vmax=1)
    ax.plot([xc], [yc], "ko", label="centroide")
    ax.set_title("arbol.png")
    ax.legend()
    plt.show()

    # plot : animation
    N = 100 # Nº frame
    d = x0.max() - x0.min()
    theta = np.linspace(0, 3*np.pi, N)
    traslacion = np.linspace(0, d, N)
    c = 3 # calidad de la imagen # c = 1 es lo mejor, con {2,3,..} mejoramos la velocidad del gif (ploteamos menos puntos) pero empeoramos la imagen
    ani = create_animation_2(x0[::c], y0[::c], z0[::c], theta, traslacion)
    if save_gif:
        ani.save("images/arbol.gif", fps = 10) 
    plt.show()

#os.chdir()
#ani.save("p4ii.gif", fps = 10)  
#os.getcwd()


if __name__ == "__main__":
    problema1(save_gif=True)
    problema2(save_gif=True)