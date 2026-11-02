# Clasificadores Naïve Bayes: las flores iris y un filtro de spam

**Versión resuelta** (para el profesor). La del alumno está en `Intro2AI/practice_bayesian_learning_v4/`.

Se programa **desde cero**, con numpy, el clasificador Naïve Bayes en sus dos variantes más usadas,
y se pone a prueba en dos problemas reales: el **gaussiano**, con las medidas de las flores **iris**,
y el **multinomial**, con un **filtro de spam** entrenado con SMS reales. Al final de cada ejercicio,
el programa comprueba que el resultado es el mismo que el de scikit-learn. El código son funciones
cortas, con un bucle por clase y otro por atributo (o por palabra), sin cálculos con matrices.

El guion está en `Intro2AI_lessons/lesson3-bayesian-learning_v4/main.tex`.

    pip install -r requirements.txt
    python 1_iris_main.py                # ejercicio 1: Naïve Bayes gaussiano, con las flores iris
    python 2_bolsa_de_palabras_main.py   # ejercicio 2: la bolsa de palabras
    python 3_spam_main.py                # ejercicio 3: Naïve Bayes multinomial, un filtro de spam
    python filtro.py                     #   e intentar engañar al filtro

| Fichero | ¿Se completa? | Contenido |
|---|---|---|
| `naive_bayes/gaussiano.py` | ejercicio 1 | `entrenar_naive_bayes` (a), `log_densidad_gaussiana` y `calcular_puntuaciones` (b) |
| `naive_bayes/bolsa_de_palabras.py` | ejercicio 2 | `construir_vocabulario` y `vectorizar` |
| `naive_bayes/multinomial.py` | ejercicio 3 | `entrenar_naive_bayes_multinomial` (a) y `calcular_puntuaciones_multinomial` (b) |
| `naive_bayes/datos.py` | no | carga de las flores y de los SMS, y partición en entrenamiento y prueba |
| `naive_bayes/evaluacion.py` | no | matriz de confusión, precisión y exhaustividad |
| `herramientas.py` | no | las comprobaciones y la figura del ejercicio 1 |
| `data/` | no | SMS Spam Collection v.1 (véase `data/README.md`) |

Los huecos están marcados con `EJERCICIO N: SE DEBE COMPLETAR EL CÓDIGO POR EL ESTUDIANTE` (en los
ejercicios 1 y 3, `EJERCICIO 1 a)`, `b)`...). Mientras no se completan, lanzan `NotImplementedError`.
Cada programa comprueba sus resultados y escribe [BIEN] o [FALLA]. Los programas son cortos y cada
apartado del guion es una función: léalos para saber qué hace y qué comprueba cada uno. La figura
del ejercicio 1 se guarda en `figures/`. scikit-learn solo se usa para cargar las flores iris y para
comparar los resultados. Hay que ejecutar los programas **desde este directorio**.
