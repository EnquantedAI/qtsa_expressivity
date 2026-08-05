# expressivity_krylov

Access the expressivity of a variational quantum circuit using the Krylov method. 

## Krylov metrics

Krylov metrics describe the characteristics of the Krylov space

```
span{v, Av, A^2v, ...}
```

The choice of the operator `A` is crucial. Krylov metrics have found success as expressivity metrics in the theory of Quantum Reseirvoir Computing and Quantum Extreme Learning when applied to the effective Hamiltonian, and can similarly be applied to a Quantum Neural Network derived from a Hamiltonian. In the general case however, QNNs don't have an obvious choice for the operator building the studied Krylov space. Previous experiments (not documented in this module) have showed that training-unaware choices of the operator fail to capture expressivity of the QNN.

Currently the Quantum Neural Tangent Kernel (QNTK) is examined as a potential candidate for the root operator. QNTK has already been studied in its own right as an expressivity metric. Krylov methods in this context can be used to bypass performance bottlenecks. 

## How to run

Runnable code is located inside `experiments.ipynb`, however it's currently in a bit of a mess. The notebook will soon be cleaned up & proper unit tests will be added to submodules.

## Structure

Code works with implicit matrices via a lightweight SymmetricOperator class (extending scipy.linalg.LinearOperator). This bypasses the need for explicitly building the matrix, instead the interface only exposes a matrix-vector product function, which is all that's required to use Krylov-based methods. The aim is to lower overhead.

`./jacobian` implements an oracle for the Jacobian-Vector Product (JVP) and Vector-Jacobian Product (VJP) using Pennylane automatic differentiation.

`./krylov` includes calculation of the desired metrics (this is mostly a wrapper of `scipy.linalg`)

`./krylov/operators` declares the abstract `SymmetricOperator` and implements `QNTKOperator` and `GramOperator` matching this interface.
