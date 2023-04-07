# TRANSFORMACIÓN ISOMÉTRICA AFÍN

## Índice 

 - [Enunciado](#id0)
 - [Introducción](#id1)
 - [Material usado](#id2)
 - [Resultados y conclusiones](#id3)
      - [Pregunta 1](#id3.1)
      - [Pregunta 2](#id3.2)
      
## Enunciado <a name=id0></a>
	
## Introducción <a name=id1></a>

La transformación isométrica afín es una operación matemática utilizada en geometría para transformar figuras geométricas en el plano. Es una transformación rígida que mantiene las distancias entre los puntos y las proporciones de las figuras, por lo que es útil para conservar la simetría y la estructura de las formas. Las transformaciones isométricas afines incluyen rotaciones, traslaciones, reflexiones y combinaciones de estas operaciones. Son herramientas importantes en la geometría, ya que permiten transformar figuras geométricas de manera precisa y eficiente.
 
## Material usado <a name=id2></a>
	
Como lenguaje de programación, se ha usado python, para realizar todo el código, predicciones y gráficas. Como material externo es necesario el archivo *arbol.png*, imagen necesaria para la realización de la segunda pregunta. En cuanto a librerías serán necesarias *matplotlib* para las gráficas y *skimage* para cargar la imagen mencionada anteriormente en python como matriz de pixeles.
	
## Resultados y conclusiones <a name=id3></a>

Para las traslaciones únicamente es necesario sumar a las componentes $x, y, z$ el valor correspondiente. Por otro lado para la rotación se usará la matriz de rotación 

$$
R = \begin{pmatrix} 
  \cos \theta & -\sin \theta & 0 \\
  \sin \theta & \cos \theta & 0 \\
  0 & 0 & 1 \\ 
\end{pmatrix}
$$

El enunciado nos pide que sea una rotación respecto al centroide de la figura. Para ello, sean $X = \lbrace x_1, \dots, x_N\rbrace$, $Y = \lbrace y_1, \dots, y_N\rbrace$, $Z = \lbrace z_1, \dots, z_N\rbrace$ tres conjuntos de puntos que componen la figura: $figura = \lbrace (x_i,y_i,z_i) \ |\ i \in \lbrace 1,\dots,N\rbrace \rbrace$. Entonces para la rotación respectiva, realizamos lo siguiente: 

 1) Definimos el centroide. $c = (xc, yc, zc)$ donde

$$ 
xc = \dfrac{1}{N}\sum_{x\in X} x, \quad yc = \dfrac{1}{N}\sum_{y\in Y} y, \quad zc = \dfrac{1}{N}\sum_{z\in Z} z
$$

 2) Trasladamos los puntos al centro (0,0). $X' = \lbrace x-xc \ |\  x\in X\rbrace$, $Y' = \lbrace y-yc \ |\  y\in Y\rbrace$, $Z' = \lbrace z-zc \ |\  z\in Z\rbrace$.
 3) Los rotamos. 

$$ 
W = \left\lbrace R \begin{pmatrix} x_i \\ y_i \\ z_i \\ \end{pmatrix} \ |\  i\in \lbrace 1,\dots, N\rbrace\ \right\rbrace 
$$

 4) Los devolvemos a su posición original. $Q = \lbrace w+c \ | w\in W\rbrace$

 
### Pregunta 1 <a name=id3.1></a>

Para generar una figura 3d aleatoria, utilizamos la función *axes3d.get_test_data()* predefinada de python. Para las rotaciones y traslaciones, hemos definido las funciones *Rotacion* y *Traslación* respectivamente. Una vez definidas y con los datos cargados podemos crear la animación. Además, para apreciar la subida de nivel en el eje $z$ de la figura, he graficado el centroide de la figura en el plano $\lbrace z=0\rbrace$ para poder tener punto de referencia fijo. Podemos observar aquí una pequeña animación de la transaformación de la figura.

<div style="align:center;">
  <image src="/images/figura_3d.png" style="width:70%; height:8cm;">
</div>
 
### Pregunta 2 <a name=id3.2></a>

La imagen es de $350\times350$ pixeles por lo que a la hora de realizar la animación tiene que plotear 122.500 puntos (aunque terminan siendo cerca de los 40.000 cuando filtramos con el verde $< 240$). En todo caso son una gran cantidad de puntos únicamente para plotear una pequeña imagen en el gráfico. En caso de plotear todos los puntos la animación va un poco lenta. Ahora bien, podemos evitar una serie de puntos (perdiendo así calidad de imagen) ploteando únicamente 1 de cada 2 puntos o incluso 1 de cada 3, el cual es el caso del gif resultado que adjunto, *arbol.gif*. Hay que tener cuidado de la distribución de los puntos en la imagen, es decir, es muy diferente quitar 1 de cada 3 puntos de forma uniformente distribuida por todo el conjunto de la figura a quitar 1/3 de la figura por una única parte, lo que afectará gravemente pues el resultado. Por otro lado, también nos piden observar el centroide del árbol, el cual se puede apreciar en la siguiente imagen.

<div style="align:center;">
  <image src="/images/figura_2d.png" style="width:40%; height:8cm;">
  <image src="/images/centroide.png" style="width:40%; height:8cm;">
</div>

Se puede observar que nuevamente he añadido el centroide en un plano más abajo de la imagen del árbol para poder seguir el movimiento de una forma más clara. Además también observar como la calidad de la imagen en el gráfico 3d (en la imagen 2d del centroide está completa) se mantiene la calidad bastante bien. Por lo que no perdemos resolución y ayudamos al procesamiento de la animación. 
 

