# Data

This directory is reserved for generated input data used by RouteIQ experiments.

The main generated dataset is:

```text
data/trip_data.csv
```

It is intentionally excluded from version control because it can be reproduced from source code.

Generate it with:

```bash
python3 -m experiments.ml.prepare_trip_data
```

The generated trip dataset contains synthetic travel observations used for the machine-learning and predict-then-optimize workflows.
