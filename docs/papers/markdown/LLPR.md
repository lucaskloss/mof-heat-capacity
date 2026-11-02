# Supplied LLPR checkpoints

This note describes the PET last-layer prediction-rigidity (LLPR) checkpoints
used by the project. It covers what the checkpoints provide and what their
uncertainty represents. The project-specific use of their member predictions
in heat-capacity and CEA calculations is documented in [CEA.md](CEA.md).

The method follows Bigi *et al.*, [“A prediction rigidity formalism for
low-cost uncertainties in trained neural networks”](https://doi.org/10.1088/2632-2153/ad805f),
and uses metatrain's [LLPR implementation](https://docs.metatensor.org/metatrain/latest/architectures/generated/llpr.html).

## What the checkpoints represent

LLPR varies the final linear energy readout while keeping the PET feature
representation fixed. For frozen last-layer features $\mathbf f(\mathbf x)$,
the central energy is $\bar V(\mathbf x)=\mathbf w_0^{\mathsf T}\mathbf f(\mathbf x)$.
The predictive variance has the form

$$\sigma_{\mathrm{LLPR}}^2(\mathbf x)=\alpha^2\mathbf f(\mathbf x)^{\mathsf T}\left(\mathbf F^{\mathsf T}\mathbf F+\lambda\mathbf I\right)^{-1}\mathbf f(\mathbf x).$$

Here $\mathbf F$ is the feature matrix used to define the covariance,
$\lambda$ is its regularizer, and $\alpha$ is the calibration scale. The
checkpoint stores the factorization and calibration parameters needed for
inference. The scalar `energy_uncertainty` output is a predictive standard
deviation in eV. It is not a variance, a trajectory sampling error, or a
standard error on a heat-capacity estimate.

The checkpoints also contain persistent last-layer member readouts
$\mathbf w_i$. They provide signed, member-resolved energies

$$V^{(i)}(\mathbf x)=\mathbf w_i^{\mathsf T}\mathbf f(\mathbf x).$$

These values preserve correlations in each member's predictions across
structures. The analysis centers the supplied member readouts on the central
PET readout, so their mean reproduces the central prediction up to numerical
precision. A finite-member energy spread should be consistent with the
analytical `energy_uncertainty`; disagreement is a convergence or
compatibility diagnostic, not proof of calibration.

LLPR uncertainty only probes directions in the final readout represented by
the stored feature covariance. It cannot reveal errors shared by the frozen
PET representation, missing chemistry, or bias in the reference labels. The
reported spread is therefore **LLPR model uncertainty**, not the uncertainty
across independently parameterized PET models or a complete measure of model
error.

## Supplied checkpoints

The project uses these calibrated 64-member PET-MAD and PET-SOL checkpoints:

```text
models/pet-mad-1.5-s_40nn_nostress-llpr.ckpt
models/pet_sol-s-best_nostress-llpr.ckpt
```

Inference provides the central `energy`, scalar `energy_uncertainty`, and
member-resolved `energy_ensemble` outputs. The analysis records the checkpoint
SHA-256 to identify the same supplied members in downstream outputs. Keep each
checkpoint paired with its matching central PET model and export.

Before interpreting its spread, consult the supplied calibration and coverage
evidence on held-out reference structures. Checkpoint compatibility and
agreement between analytical and finite-member uncertainty do not establish
that the calibration data cover every MOF-5 loading, temperature, adsorption
environment, or framework distortion of interest.

For the complete analysis sequence, member persistence requirements, CEA
validity diagnostics, heat-capacity error propagation, and output fields, see
[CEA.md](CEA.md). For commands and file locations, see the
[property workflow guide](../scripts/properties/README.md).
