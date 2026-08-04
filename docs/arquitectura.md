# Decisiones de arquitectura

## Limpieza del dataset

### Registros sin rating

Se eliminarán únicamente los registros cuyo valor de `rating` sea nulo.

**Justificación**

- `rating` es la variable objetivo.
- Un modelo supervisado necesita conocer la etiqueta correcta durante el entrenamiento.
- Solo existen 4 registros con este problema, por lo que la pérdida de información es mínima.

---

### Valores nulos en country

Los países nulos serán reemplazados por `Unknown`.

**Justificación**

Eliminar esos registros implicaría perder más de 800 observaciones.

---

### Valores múltiples

Las columnas `country` y `listed_in` contienen múltiples valores separados por comas.

Para simplificar el problema se utilizará únicamente el primer valor.

Ejemplo:

United States, Canada

↓

United States