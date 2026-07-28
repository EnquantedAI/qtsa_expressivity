# GQNN

`GQNN` is a PennyLane quantum neural network wrapper designed for PyTorch training and quantum-state analysis.

It creates one shared variational circuit with two outputs:

- `gqnn.qlayers(inputs)` returns Pauli-Z expectation values for training.
- `gqnn.state_circuit(inputs, weights)` returns the complete statevector for analysis, including QFIM calculations.

The trainable circuit uses PennyLane’s `StronglyEntanglingLayers`; its weight tensor has shape:

```python
(n_layers, n_qubits, 3)
```

## Circuit structure

For an input vector `inputs`, the circuit performs:

1. Data embedding through a feature map.
2. A trainable `StronglyEntanglingLayers` ansatz.
3. Either Pauli-Z expectation-value measurements or a full statevector measurement.

The available feature maps are:

| `fm_style` | Feature map |
|---|---|
| `"zzfm"` | Custom second-order Havlíček ZZ feature map |
| `"iqp"` | PennyLane `IQPEmbedding` |
| `"X"` | `AngleEmbedding` with X rotations |
| `"Y"` | `AngleEmbedding` with Y rotations |
| `"Z"` | `AngleEmbedding` with Z rotations |

## Initialization

```python
import pennylane as qml

n_qubits = 4
n_layers = 2

quantum_device = qml.device(
    "default.qubit",
    wires=n_qubits,
)

gqnn = GQNN(
    n_layers=n_layers,
    n_qubits=n_qubits,
    quantum_device=quantum_device,
    interface="torch",
    diff_method="best",
    fm_style="zzfm",
    meas=,
)
```

`quantum_device` selects the PennyLane backend. It can be a simulator such as `"default.qubit"` or a supported quantum-hardware backend.

## Important attributes

```python
gqnn.n_layers
gqnn.n_qubits
gqnn.quantum_device
gqnn.weight_shape
gqnn.parameter_count
gqnn.circuit
gqnn.state_circuit
gqnn.qlayers
```

`gqnn.qlayers` is the PyTorch-compatible quantum layer. Its trainable parameters are stored in:

```python
gqnn.qlayers.weights
```

For example:

```python
print(gqnn.qlayers.weights.shape)
print(gqnn.parameter_count)
```

## Training with PyTorch

Pass a feature tensor with one value per qubit to `gqnn.qlayers`.

```python
import torch

inputs = torch.tensor(
    [0.1, 0.2, 0.3, 0.4],
    dtype=torch.float32,
)

output = gqnn.qlayers(inputs)
print(output)
```

With `meas=[0]`, `output` contains the expectation value:

```python
<Z_0>
```

For several measurement wires:

```python
gqnn = GQNN(
    n_layers=2,
    n_qubits=4,
    quantum_device=quantum_device,
    meas=,[1][2][3]
)

output = gqnn.qlayers(inputs)
```

The output then contains one Pauli-Z expectation value per specified wire.

### Optimization example

```python
optimizer = torch.optim.Adam(
    gqnn.qlayers.parameters(),
    lr=0.01,
)

target = torch.tensor([1.0])

for _ in range(100):
    optimizer.zero_grad()

    prediction = gqnn.qlayers(inputs)
    loss = torch.mean((prediction - target) ** 2)

    loss.backward()
    optimizer.step()
```

After `optimizer.step()`, the optimized variational parameters are held in:

```python
trained_weights = gqnn.qlayers.weights
```

## Statevector simulation

`gqnn.circuit` returns expectation values, so it is intended for ordinary model inference and training.

```python
expectations = gqnn.circuit(
    inputs,
    gqnn.qlayers.weights,
)
```

`gqnn.state_circuit` returns the full quantum statevector instead:

```python
state = gqnn.state_circuit(
    inputs,
    gqnn.qlayers.weights,
)

print(state.shape)
```

For `n_qubits`, the statevector contains $2^{\text{n_qubits}}$ complex amplitudes.

```python
print(state)
```

## QFIM state function

`state_function` evaluates the statevector for an explicit parameter vector.

```python
state = gqnn.state_function(
    inputs=inputs,
    flat_parameters=gqnn.qlayers.weights,
)
```

The supplied parameters are flattened internally, validated, reshaped to `gqnn.weight_shape`, and passed to `state_circuit`.

The term `flat_parameters` means a one-dimensional representation of every trainable circuit parameter:

```python
flat_parameters = gqnn.qlayers.weights.reshape(-1)
```

It contains only ansatz weights. It does not contain feature-map inputs.

```python
state = gqnn.state_function(
    inputs=inputs,
    flat_parameters=flat_parameters,
)
```

This explicit-parameter interface is useful for QFIM code because it can evaluate the state at arbitrary parameter values near the trained point.

## Trained QFIM point

To evaluate a QFIM at the learned parameters:

```python
trained_parameters = gqnn.qlayers.weights

state = gqnn.state_function(
    inputs=inputs,
    flat_parameters=trained_parameters,
)
```

Do not use `.detach()`, `.numpy()`, or `pennylane.numpy.asarray()` if the QFIM implementation differentiates through `flat_parameters`. Those conversions can disconnect the computation from PyTorch autograd.

## Drawing the circuit

`draw_circuit` returns a function that accepts the QNode arguments.

```python
draw = gqnn.draw_circuit(
    title="GQNN circuit",
    style="pennylane",
    decimals=2,
)

draw(inputs, gqnn.qlayers.weights)
```

This displays the feature map, variational layers, and Pauli-Z measurements.

## Parameters vs inputs

| Value | Meaning | Typical shape |
|---|---|---|
| `inputs` | Classical data encoded into the feature map | `(n_qubits,)` |
| `weights` | Trainable variational circuit parameters | `(n_layers, n_qubits, 3)` |
| `flat_parameters` | The same weights in one dimension | `(parameter_count,)` |
| `parameter_count` | Number of scalar trainable parameters | `n_layers * n_qubits * 3` |

The normal PyTorch call is:

```python
output = gqnn.qlayers(inputs)
```

The equivalent explicit QNode call is:

```python
output = gqnn.circuit(
    inputs,
    gqnn.qlayers.weights,
)
```

The statevector call is:

```python
state = gqnn.state_circuit(
    inputs,
    gqnn.qlayers.weights,
)
```