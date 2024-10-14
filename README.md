# Motor de Redes Neuronales

Este proyecto consiste en la implementación de un motor de redes neuronales desde cero, optimizado sin el uso de frameworks como PyTorch o TensorFlow, utilizando únicamente Python y bibliotecas básicas como `numpy`. La memoria y los métodos desarrollados están detallados en la documentación adjunta.

## Tabla de Contenidos
1. [Descripción](#descripción)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Experimentos y Resultados](#experimentos-y-resultados)
6. [Contribuciones](#contribuciones)

## Descripción

Este proyecto se centra en la creación de un motor de redes neuronales utilizando principios de optimización y heurística. El objetivo es implementar una red neuronal desde cero y optimizar su rendimiento sin depender de bibliotecas externas avanzadas para redes neuronales. El motor es modular y permite reutilizar y generalizar los métodos implementados.

## Estructura del Proyecto

El proyecto está organizado en las siguientes secciones:

- **Memoria:** Documentación detallada sobre el proyecto, que incluye el desarrollo teórico y la justificación de los métodos empleados.
- **Código:** Implementación de los métodos descritos, dividida en módulos según la funcionalidad (optimización, entrenamiento, evaluación).
- **Experimentos:** Conjuntos de pruebas para validar la efectividad de la red neuronal implementada.

## Instalación

1. Clona este repositorio en tu máquina local:

    ```bash
    git clone https://github.com/usuario/motor-redes-neuronales.git
    ```

2. Instala las dependencias requeridas:

    ```bash
    pip install numpy
    ```

3. (Opcional) Crea un entorno virtual para aislar las dependencias del proyecto:

    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

## Uso

Puedes ejecutar el código en **Jupyter Notebooks** para facilitar la interacción y la documentación del proceso de entrenamiento y evaluación de la red neuronal.

1. Abre el entorno Jupyter:

    ```bash
    jupyter notebook
    ```

2. Dirígete al notebook principal `main.ipynb` donde se documentan los experimentos y resultados obtenidos.

3. Modifica los parámetros de la red neuronal (número de capas, tasa de aprendizaje, etc.) para ejecutar tus propias pruebas.

## Experimentos y Resultados

## Planes Futuros
- Añadir opción para agregar n hidden layers
- Añadir más funciones de activación
- Añadir más funciones de optimización
- Mejorar el diseño de la clase core para que admita funciones de activación y optimización como parámetros


## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas contribuir, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama (`git checkout -b mejora-característica`).
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva característica'`).
4. Haz push a la rama (`git push origin mejora-característica`).
5. Abre un **Pull Request**.
