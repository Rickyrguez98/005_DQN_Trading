# Sistema de Trading DQN

Descripción del Proyecto
Este proyecto aprovecha las Redes Q Profundas (DQN, por sus siglas en inglés) para desarrollar un sistema de trading basado en los datos históricos de precios en intervalos de 5 minutos de las acciones de Apple. Se enfoca en automatizar decisiones de trading y optimizar estrategias de trading a través del aprendizaje por refuerzo.

## Características

- Trading Automatizado en Acciones de Apple: Entrena con los datos de acciones de Apple, utilizando movimientos de precios en intervalos de 5 minutos para prever y ejecutar operaciones.
- Toma de Decisiones Secuenciales: Modela escenarios de trading realistas donde la acción de cada agente progresa en el tiempo por un paso de tiempo.
- Estrategia Basada en Datos: Utiliza técnicas avanzadas de aprendizaje automático para aprender de datos históricos e identificar oportunidades de trading rentables.
- Preparación de Datos
El modelo de trading utiliza un conjunto de datos meticulosamente preparado derivado de los precios de las acciones de Apple:

- Ingeniería de Características: Los datos históricos de precios se transforman para incluir características rezagadas, capturando tendencias y momentum en intervalos secuenciales de 5 minutos.
- Indicadores Técnicos: Implementa indicadores de análisis técnico como el Índice de Fuerza Relativa (RSI) para proporcionar información sobre el momentum del precio de la acción y condiciones que podrían estar sobrecompradas o sobrevendidas.

## Pila Tecnológica

- Python: Lenguaje de programación principal.
- TensorFlow/Keras: Utilizados para construir y entrenar el modelo DQN.
- NumPy/Pandas: Para manipulación y preparación de datos.
- Matplotlib: Para visualizar resultados de trading y métricas de rendimiento.









