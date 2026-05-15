"""Classical machine-learning teaching nodes.

Two parallel tracks:

- ``Edu*`` — hand-written implementations that expose the core math
  (distances, normal equation, gradient descent) for step-tracing.
  Right for ≤200-row teaching datasets.
- production wrappers (``KNN``, ``LinearRegression``,
  ``LogisticRegression``, ``SVMClassifier``, ``DecisionTreeClassifier``)
  — sklearn-backed, drop-in replacements that scale and add the full
  knob set. Right for "the textbook is done, ship it."
"""
