# Publicar la versión para los alumnos

Este repositorio (`4rtur1t0/Intro2AI_full`, privado) contiene el material completo,
incluidas las soluciones. Los alumnos solo ven el repositorio público
[`4rtur1t0/Intro2AI`](https://github.com/4rtur1t0/Intro2AI).

## Ramas

| Rama      | Contenido                                                        |
|-----------|------------------------------------------------------------------|
| `develop` | Trabajo en curso.                                                |
| `main`    | Versión completa para profesores (con `solutions/` y `utils/`).  |
| `public`  | Versión para alumnos: igual que `main`, pero **sin** `solutions/`, `utils/` ni este fichero. |

```
develop ──merge──▶ main ──merge──▶ public ──push──▶ GitHub Action ──▶ Intro2AI (alumnos)
                                     │
                                     └─ antes del push: borrar solutions/ y utils/
```

**Hacer push a `public` es publicar.** La Action
[.github/workflows/sync-main.yml](.github/workflows/sync-main.yml) se ejecuta en cada
push a `public` y hace `git push public public:main --force` sobre el repositorio
de los alumnos. No hay paso de revisión intermedio.

## Pasos

### 0. Actualizar el repositorio local

```bash
git fetch origin
```

Cada rama local se actualiza con `git pull` en el paso que le toca.

### 1. Subir los cambios a `develop`

```bash
git checkout develop
git pull
# ... editar y probar ...
git add <ficheros>
git commit -m "Descripción del cambio"
git push origin develop
```

### 2. Llevar `develop` a `main`

```bash
git checkout main
git pull
git merge develop        # normalmente es fast-forward
git push origin main
```

Al terminar, `develop` y `main` deben apuntar al mismo commit.

> En el histórico también aparece el camino inverso: se hace commit directamente en
> `main` y luego se pone `develop` al día con `git checkout develop && git merge main`.
> Cualquiera de las dos formas vale si las dos ramas acaban iguales.

### 3. Llevar `main` a `public` y quitar lo privado

```bash
git checkout public
git pull
git merge --no-ff --no-commit main
git rm -r -q -f --ignore-unmatch solutions utils PUBLICACION.md
```

- `--no-commit` hace que el merge se detenga antes de crear el commit, para borrar
  las carpetas privadas en el mismo commit del merge.
- Si en `main` se han modificado ficheros de `solutions/`, git avisará de un conflicto
  `CONFLICT (modify/delete)`. Es lo esperado: el `git rm` de arriba lo resuelve
  (el `-f` es necesario precisamente por eso).
- Si hay conflictos en otros ficheros (material de los alumnos), resuélvelos a mano y
  márcalos con `git add <fichero>`.

Luego crea el commit:

```bash
git commit -m "Publicación: <qué se publica>"
```

> En el histórico esto se hizo en dos commits: primero `Merge branch 'main' into public`
> y después otro commit que borraba `solutions/` y `utils/` (los commits "Added actions"
> o "Added Dijkstra" de `public`). El resultado final es el mismo, pero el merge
> intermedio contiene las soluciones. Con `--no-commit` ese commit intermedio no existe.

### 4. Comprobar antes de publicar

```bash
# No debe imprimir nada:
git ls-tree -r --name-only HEAD | grep -E '^(solutions|utils)/|^PUBLICACION\.md$'

# Revisar qué va a cambiar para los alumnos:
git diff --stat origin/public HEAD
```

Si hace falta, ejecuta el código de los alumnos desde esta rama para comprobar que
funciona sin `solutions/`.

### 5. Publicar

```bash
git push origin public
```

Después:

1. En la pestaña **Actions** de `Intro2AI_full`, comprueba que el workflow
   *"Sync Public Branch (in this private repository) to Public Repo"* termina en verde.
2. Abre <https://github.com/4rtur1t0/Intro2AI> y comprueba que están los cambios y que
   **no** aparecen `solutions/` ni `utils/`.

### 6. Volver a la rama de trabajo

```bash
git checkout develop
```

## Chuleta

```bash
git fetch origin

git checkout develop && git pull
# ... commit de los cambios ...
git push origin develop

git checkout main && git pull
git merge develop
git push origin main

git checkout public && git pull
git merge --no-ff --no-commit main
git rm -r -q -f --ignore-unmatch solutions utils PUBLICACION.md
git commit -m "Publicación: <qué se publica>"
git ls-tree -r --name-only HEAD | grep -E '^(solutions|utils)/|^PUBLICACION\.md$'   # debe salir vacío
git push origin public      # ← esto publica a los alumnos

git checkout develop
```

## ⚠️ Las soluciones están en el historial del repositorio público

La Action sube **todo el historial** de `public`, no solo la última versión. Como
`public` se actualiza haciendo merge de `main`, los commits de `main`, que sí contienen
`solutions/`, forman parte de ese historial. Por ejemplo, en `4rtur1t0/Intro2AI` los
commits `6bad186` y `8bbbb9b` contienen `solutions/` completo. Un alumno puede
recuperarlas con:

```bash
git log --all -- solutions
git checkout 8bbbb9b -- solutions
```

Borrar `solutions/` en el último commit (como hace el paso 3) solo las quita de la
versión actual, no del historial.

Para evitarlo en adelante, el paso 3 se puede cambiar para que publique una "foto" de
`main` sin enlazar su historial:

```bash
git checkout public
git read-tree -u --reset main        # deja el árbol de trabajo igual que main
git rm -r -q -f --ignore-unmatch solutions utils PUBLICACION.md
git commit -m "Publicación: <qué se publica>"
```

El commit resultante solo tiene como padre el `public` anterior, así que los commits de
`main` no llegan al repositorio público. Aun así, esto no borra lo que ya está publicado:
para limpiar el historial existente habría que reescribir `public` (por ejemplo, empezar
una rama huérfana con `git checkout --orphan`) y hacer un force-push. Los alumnos que ya
tengan el repositorio clonado tendrían que volver a clonarlo.
