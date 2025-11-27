# Predator–Prey Neural Evolution (Love2D)

**Modelo entrenado disponible:** generación **2568**
**Archivo de estado:** `sim_state.lua`
*(Debe colocarse donde `love.filesystem` pueda leerlo — normalmente en la raíz del proyecto o en `%APPDATA%/LOVE/<project>/`.)*

---

## Resumen

Este proyecto implementa una simulación evolutiva de depredadores y presas usando:

* **Redes neuronales feed-forward** (1 capa oculta)
* **Algoritmo genético** con mutación (el crossover está definido pero no se usa en la versión actual)
* **Entrenamiento por episodios** (supervivencia/capturas)
* **Visualización en tiempo real** mediante **Love2D**

Cada agente (presa o depredador) usa una red neuronal para decidir dirección, aceleración e intención. Al final de cada episodio, los puntajes determinan un campeón cuyo genoma se clona para formar la siguiente generación, con mutación opcional.

---

##  Quick Start

1. Instala **Love2D** → [https://love2d.org](https://love2d.org)
2. Coloca todos los `.lua` del proyecto en una carpeta.
3. Ejecuta:

   ```
   love .
   ```
4. (Opcional) Coloca `sim_state.lua` con `gen = 2568` en la carpeta del proyecto.
5. Controles principales:

   * **P** → Pausar / Reanudar
   * **S** → Guardar estado en `sim_state.lua`
   * **L** → Cargar estado desde `sim_state.lua`
   * **M** → Activar/Desactivar mutación
   * **, / .** → Ajustar velocidad
   * **O** → Saltar 10 generaciones
   * **+ / -** → Añadir/Quitar presas
   * **] / [** → Añadir/Quitar depredadores

---

##  Configuración principal (tabla `cfg`)

| Parámetro           | Valor    | Descripción                                           |
| ------------------- | -------- | ----------------------------------------------------- |
| `prey_count`        | 100      | Presas iniciales                                      |
| `pred_count`        | 15       | Depredadores iniciales                                |
| `episode_time`      | 300      | Duración del episodio (s)                             |
| `vel_base_prey`     | 120      | Velocidad base presa                                  |
| `vel_base_pred`     | 160      | Velocidad base depredador                             |
| `stamina_max_prey`  | 1.0      | Stamina máxima presa                                  |
| `stamina_max_pred`  | 1.0      | Stamina máxima depredador                             |
| `drain_prey`        | 0.12     | Drenaje de stamina presa                              |
| `drain_pred`        | 0.18     | Drenaje depredador                                    |
| `recover_prey`      | 0.06     | Recuperación presa                                    |
| `recover_pred`      | 0.04     | Recuperación depredador                               |
| `vision_prey`       | 1000     | Radio de visión de presas                             |
| `vision_pred`       | 150      | Radio de visión de depredadores                       |
| `pmut`              | 0.05     | Probabilidad de mutación por peso/bias                |
| `pcross`            | 0.5      | Prob. de usar mezcla de pesos (no usado en GA actual) |
| `H`                 | 8        | Neuronas ocultas                                      |
| `input_size`        | 9        | Entradas de la red                                    |
| `output_size`       | 3        | Salidas                                               |
| `capture_radius`    | 12       | Distancia para captura                                |
| `world_w / world_h` | 1200×800 | Mundo                                                 |

---

##  Arquitectura de la Red Neuronal

Cada agente usa una red:

```
Inputs (9)
   ↓
Hidden layer (H = 8, tanh)
   ↓
Outputs (3: steer, thrust, intent)
```

Entradas típicas incluyen:

* dx/dy hacia objetivo
* distancia normalizada
* ángulo relativo
* densidad local
* stamina del agente
* stamina promedio de vecinos
* indicador binario de si está siendo perseguido

Las salidas controlan:

* Ángulo de giro
* Fuerza de aceleración
* Intención (0–1) que influye en decisiones internas

---

## Comportamiento de Agentes

* **Presas**: intentan alejarse de depredadores, conservar stamina y sobrevivir.
* **Depredadores**: buscan la presa más cercana dentro de visión.
* La stamina afecta velocidad → sprint rápido pero drena.

Capturas:

* Si un depredador está a distancia `< capture_radius` → la presa muere y se incrementa `metrics.captured` del depredador.

---

##  Episodios y Fitness

Cada episodio dura `episode_time` segundos **o** termina cuando se capturan todas las presas.

**Fitness:**

* **Presas** → supervivencia + bonus por cada *esquive* registrado.
* **Depredadores** → número de capturas + bonus por eficiencia.

---

## Algoritmo Genético

Implementación actual (en `Sim:runGA()`):

1. Se elige un **campeón** (mayor fitness) por especie.
2. Toda la siguiente generación se construye clonando ese campeón.
3. Si `pmut > 0`, se aplica mutación gaussiana a los pesos.
4. *Nota:* El crossover existe en `NN.crossover` pero no se usa aquí.

Esto produce evolución pero baja diversidad; puede mejorarse fácilmente.

---

##  Guardado y Carga (`sim_state.lua`)

Formato:

```lua
return {
  gen = 2568,
  prey_genomes = {
    { wh = {...}, bh = {...}, wo = {...}, bo = {...} },
    ...
  },
  pred_genomes = {
    { ... },
    ...
  }
}
```

### Para cargar la generación **2568**

1. Coloca `sim_state.lua` en la carpeta del proyecto.
2. Ejecuta `love .`
3. Presiona **L** → los genomas y la generación se restauran.

---

## Bugs conocidos / Cosas a mejorar

* `pcross`, `elitism`, `tournament_k` existen pero no se usan.
* `best_fitness` se muestra pero no se actualiza correctamente.
* Mutación ON/OFF pisa el valor original de `pmut`.
* El GA puede volverse demasiado homogéneo (solo clonación del campeón).
* No se guarda `randomseed`, por lo que la reproducción exacta del entrenamiento no está garantizada.

---

## Sugerencias de mejoras

* Implementar selección por torneo real o ruleta.
* Usar crossover (`NN.crossover`) en reemplazo.
* Registrar y visualizar curva de fitness por generación.
* Guardar la semilla RNG dentro de `sim_state.lua`.
* Añadir `conf.lua` con `identity` fijo para ajustar carpeta de guardado.

---

## Archivo mínimo `sim_state.lua` para fijar la generación 2568

```lua
return {
  gen = 2568,
  prey_genomes = {},
  pred_genomes = {}
}
```