# 004_DL

## Proyecto 4 de Micro


En este proyecto nos concentramos en la aplicación de tres potentes modelos de redes neuronales: Deep Neural Networks (DNN), Recurrent Neural Networks (RNN) y Convolutional Neural Networks (CNN). Nuestro objetivo es ir más allá de simplemente predecir precios de acciones, centrándonos en la clasificación de señales de compra y venta utilizando datos temporales variados. Este enfoque nos permite anticipar con precisión si el precio de una acción estará dentro de un rango específico, proporcionando así una herramienta estratégica para maximizar la ganancia esperada.

## Descripción de modelos


1. Deep Neural Network (DNN): Una DNN es una red neuronal profunda utilizada para clasificación y regresión. Aprende representaciones complejas de los datos y puede predecir probabilidades de clase en función de características de entrada. Ampliamente utilizada en visión por computadora, procesamiento de lenguaje natural y otras aplicaciones de aprendizaje profundo.


2. Recurrent Neural Network (RNN): Una RNN es una red neuronal especializada en datos secuenciales. Utiliza conexiones recurrentes para modelar dependencias temporales y realizar predicciones basadas en el contexto acumulado a lo largo de la secuencia. Es esencial para tareas como traducción automática, análisis de sentimientos y series temporales.


3. Convolutional Neural Network (CNN): Una CNN es una red neuronal diseñada para procesar datos espaciales, como imágenes. Utiliza filtros convolucionales para extraer características locales y aprender representaciones jerárquicas de las imágenes. Ampliamente utilizada en reconocimiento de objetos, segmentación de imágenes y otras aplicaciones de visión por computadora.


## Datos de entrada y salida


1. Para nuestras X o datos de salida usamos el precio Close, x_t-n en este caso hasta -3 y el RSI con ventana de 28.
2. Para la Y o datos de salida usamos una condicion para ver si el precio xt era menor para determinar la compra y menor la venta dependiendo del quinto precio en los datos (P_t+5)

### Parámetros por operación

1. Stop Loss: Este parámetro establece el nivel de precio al cual se activará una orden de venta para cerrar una posición larga y limitar las pérdidas. (0.00250,0.05)
2. Take Profit: Especifica el nivel de precio al cual se activará una orden de venta para cerrar una posición larga y asegurar las ganancias. (0.00250, 0.05)
2. Número de Acciones (n_shares): Este parámetro define la cantidad de acciones a comprar o vender en cada operación. (5, 200)

### Parámetros por función:

1. Deep Neural Network (DNN): 
* Number of units: Se refiere a la cantidad de neuronas o nodos en una capa específica de una red neuronal. (50,200)
* Number of layers: Se refiere a la profundidad de la red neuronal. (1,3)
* LR: Controla qué tan grande son los pasos que el algoritmo de optimización toma durante el proceso de entrenamiento. (1e-4, 1e-2)
* Activation functions: Las funciones de activación se utilizan en cada neurona de una red neuronal para introducir no linealidades en el modelo. ("relu")

2. Recurrent Neural Network (RNN): 
* Number of units: Se refiere a la cantidad de neuronas o nodos en una capa específica de una red neuronal. (20,100)
* LR: Controla qué tan grande son los pasos que el algoritmo de optimización toma durante el proceso de entrenamiento. (1e-4, 1e-2)

3. Convolutional Neural Network (CNN):
* Number of layers: Se refiere a la profundidad de la red neuronal. (1,3)
* Filters: Son matrices pequeñas que se deslizan sobre la entrada de una capa convolucional. (32, 128)
* Kernel size: Es la dimensión espacial del filtro que se utiliza en una operación de convolución. (2, 5)
* Pool size: Reduce la dimensionalidad de las características convolucionales al agrupar las salidas de múltiples neuronas. (1, 2)
* Strides: Determinan el desplazamiento del filtro durante la operación de convolución. (1, 2)
* Padding: Se utiliza para controlar el tamaño de la salida después de aplicar operaciones de convolución. ('valid', 'same')








