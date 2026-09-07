# Point Edge Transformers and their role in this project

## Article and scope

Sergey N. Pozdnyakov and Michele Ceriotti, *Smooth, exact rotational
symmetrization for deep learning on point clouds*, NeurIPS 2023. The supplied
[PET.pdf](PET.pdf) is the 33-page arXiv version 3, dated 6 February 2024.
The [conference record](https://proceedings.neurips.cc/paper_files/paper/2023/hash/fb4a7e3522363907b26a86cc5be627ac-Abstract-Conference.html)
and [arXiv record](https://arxiv.org/abs/2305.19302) identify the publication.
Page references below refer to the supplied PDF.

The paper introduces two distinct contributions: **PET**, a transformer-based
architecture for atomistic prediction, and **ECSE** (Equivariant Coordinate
System Ensemble), a procedure that makes an otherwise rotationally
unconstrained model exactly rotation equivariant. PET is the neural-network
architecture; PET-MAD is a later pretrained potential built using PET. The
paper's benchmark models are separate fits, not the checkpoints used in this
MOF-5 repository. The final section connects the paper to the local code and
model metadata.

## Scientific motivation

An MLIP replaces repeated electronic-structure calculations by a learned map
from atomic species, positions, and the periodic cell to potential energy.
An extensive energy can be written as a sum of learned local contributions:

$$U_\theta(\mathbf R,\mathbf h,\mathbf Z)=\sum_i b_{Z_i}+\sum_i\varepsilon_i(\mathcal A_i;\theta).$$

The species-dependent baselines $b_Z$ are constant at fixed composition.
Forces follow from $\mathbf F_i=-\partial U_\theta/\partial\mathbf R_i$.
Differentiating this same scalar energy gives conservative forces, but does
not by itself guarantee rotational symmetry or accuracy.

The paper asks whether exact rotational symmetry needs to constrain every
operation inside a neural network. Distance-only message passing can fail to
distinguish some geometries. More expressive invariant or equivariant models
address this, but constrain the available operations. PET instead uses full
Cartesian geometry and an ordinary transformer; ECSE supplies exact rotational
symmetry separately. This is an architectural alternative, not a proof that
equivariant networks are generally less accurate. (Sections 1–4, pp. 1–4.)

## How the original PET architecture works

### Local environments and directed messages

Each atom $i$ has a neighborhood within a cutoff $R_c$. In a periodic system,
the displacement to a neighbor image is

$$\mathbf r_{ij\mathbf n}=\mathbf R_j+\mathbf h\mathbf n-\mathbf R_i,\qquad \lVert\mathbf r_{ij\mathbf n}\rVert<R_c.$$

PET maintains a separate feature vector for each directed edge. The message
from $i$ to $j$ need not equal the message from $j$ to $i$. All messages arriving
at one atom are processed together, but their identities are retained in
separate output tokens. This avoids compressing the whole environment into a
single atom feature before constructing outgoing messages. Attention still
uses weighted sums internally: the paper's description of aggregation-free
message passing refers to retaining distinct edge messages, not to removing
all summation. (Section 6 and Figure 2, pp. 7–8.)

### Encoding species, geometry, and incoming messages

Write $\mathbf m_{j\to i}^{(\ell-1)}$ for an incoming edge message. In the
original formulation, each neighbor token combines three sources of
information:

$$\mathbf p_{ij}^{(\ell)}=\operatorname{SiLU}\!\left(\mathbf W_r^{(\ell)}\mathbf r_{ij}+\mathbf b_r^{(\ell)}\right),\qquad \mathbf t_{ij}^{(\ell)}=M_\ell\!\left[\mathbf m_{j\to i}^{(\ell-1)}\Vert E_n^{(\ell)}(Z_j)\Vert\mathbf p_{ij}^{(\ell)}\right].$$

Each ingredient has width $d_{\mathrm{PET}}$; a multilayer perceptron (MLP)
compresses their concatenation from $3d_{\mathrm{PET}}$ to
$d_{\mathrm{PET}}$. The first block has no incoming message and therefore
compresses $2d_{\mathrm{PET}}$ inputs. A **separate central-atom token** is
$\mathbf t_{i0}^{(\ell)}=E_c^{(\ell)}(Z_i)$. Central and neighbor species
embeddings are distinct. The Cartesian embedding is a learned feature vector,
not a physical vector with a prescribed rotation law. Appendix A also describes
variants, including initializing messages with species embeddings instead of
concatenating species at every block. (Appendix A, p. 18, Eq. 5.)

### Attention inside an environment

The central token and neighbor tokens enter a transformer. For one head,
queries, keys, and values are learned linear projections of the tokens.
With $a,b$ indexing tokens in the same environment, a schematic expression is

$$\alpha_{ab}=\frac{\exp(\mathbf q_a^\mathsf T\mathbf k_b/\sqrt{d_k})\,c_b}{\sum_c\exp(\mathbf q_a^\mathsf T\mathbf k_c/\sqrt{d_k})\,c_c},\qquad \mathbf y_a=\sum_b\alpha_{ab}\mathbf v_b.$$

Here $c_0=1$ for the central token and $c_j=f_c(r_{ij})$ for a neighbor.
The cutoff multiplies the attention weights **before renormalization**, so a
departing neighbor also disappears smoothly from the normalization. Multiple
heads, feed-forward transformations, and residual connections provide the
transformer's flexibility. There is no positional encoding based on arbitrary
neighbor ordering: permuting neighbors permutes their output tokens.

Attention lets one neighbor's representation depend on the other neighbors,
so a single message-passing block can learn angular and higher-body structure
without explicitly prescribing bond-angle or spherical-harmonic features.
More attention layers refine interactions inside that environment; more
message-passing blocks propagate information between environments and enlarge
the receptive field. These are different architecture parameters. Standard
local attention has a quadratic cost in the number of local tokens, while
the overall cost remains approximately linear in atom count for fixed
neighborhood size and architecture. (Section 6, pp. 7–8; the scaling follows
from the local attention operation.)

### Outgoing messages and energy readout

Output neighbor tokens supply outgoing messages, with residual additions from
the preceding messages. Each block also has separate central and neighbor
readout heads. For the summed-edge version in Appendix A, the energy has the
form

$$U_\theta=\sum_i b_{Z_i}+\sum_{\ell=1}^{L}\sum_i\left[H_c^{(\ell)}(\mathbf x_{i0}^{(\ell)})+\sum_{j\in\mathcal A_i}f_c(r_{ij})H_n^{(\ell)}(\mathbf x_{ij}^{(\ell)})\right].$$

The edge term is cutoff weighted for smooth entry and exit of neighbors.
Contributions are collected across blocks, not only from the final block.
Appendix A also allows a cutoff-weighted **average** of edge predictions.
These learned edge contributions are not physical pair potentials: their
tokens depend on many atoms, and the directed-edge convention is absorbed in
the trained model. (Appendix A, p. 18, Eqs. 6–7.)

## Symmetry and the separate ECSE construction

Relative coordinates give translation invariance, and shared operations plus
symmetric readout give permutation invariance. Smooth activations and cutoff
handling give smooth geometry dependence. The unconstrained Cartesian
backbone does **not** enforce exact rotational invariance. Randomly rotating
training structures encourages approximate invariance but does not prove it.

ECSE constructs local coordinate frames from ordered pairs of noncollinear
neighbors, evaluates the backbone in those frames, and combines its outputs.
For a scalar local energy,

$$\varepsilon_i^{\mathrm{ECSE}}=\frac{\sum_{j\ne k}w_{ijk}\,\varepsilon_i^0(Q_{ijk}^{\mathsf T}\mathcal A_i)}{\sum_{j\ne k}w_{ijk}}.$$

The frame matrix $Q_{ijk}$ rotates with the environment, so coordinates
expressed in that frame are unchanged by an overall rotation. For vector or
tensor targets, the predicted components must be transformed back to the
original frame before averaging. The construction in Appendix F.1 uses proper
rotations, $SO(3)$; reflection invariance is an additional requirement and
should not be inferred solely from this formula.

Choosing only the closest pair would cause abrupt frame changes when neighbor
ordering changes. ECSE instead weights frames smoothly:

$$w_{ijk}=f_c(r_{ij})f_c(r_{ik})q_c\!\left(\lVert\widehat{\mathbf r}_{ij}\times\widehat{\mathbf r}_{ik}\rVert^2\right).$$

Radial factors suppress frames near the boundary, and the angular factor
suppresses ill-defined, nearly collinear frames. Fully collinear environments
need special handling; the COLL experiments use an auxiliary internally
invariant model. An adaptive inner radius and smooth pruning reduce the
number of frames. This radius controls the **frame ensemble**, and is distinct
from the adaptive neighbor selection in later PET implementations.
(Section 5, pp. 4–6; Appendices F.1–F.5, pp. 24–27.)

Forces must differentiate the **complete symmetrized energy**, including
coordinate-frame and weight derivatives. Averaging rotated backbone forces
alone omits these terms. Tight frame selection can produce large derivatives
despite mathematical smoothness, making the choice particularly consequential
for Hessians. The paper's proof-of-principle ECSE implementation adds roughly
three orders of magnitude to backbone inference cost; this is an implementation
result, not an unavoidable cost of PET. (Section 6, p. 8; Appendices F.6–F.7
and F.10–F.11, pp. 27–30.)

## Training and reported evidence

The authors subtract fitted species self-contributions, train with Adam and
random rotational augmentation, and restore the self-contributions for
inference. Their energy–force loss normalizes the two errors by moving
validation-set mean squared errors. Most experiments use token width 128,
three message-passing blocks, three attention layers per block, four heads,
SiLU, and feed-forward width 512, with dataset-specific exceptions. These are
the original paper's settings, not universal PET constants. (Appendix B,
p. 19.)

The results show strong accuracy across several problems, with identifiable
limits:

| Benchmark | Finding | Source in supplied PDF |
| --- | --- | --- |
| Liquid water | Increasing local transformer depth can outperform adding more message-passing blocks at fixed total attention-layer count; the reported force MAE reaches $14.4\ \mathrm{meV\,\mathring A^{-1}}$ at a $4.25\ \mathring A$ cutoff. | Section 7, pp. 8–9; Appendix C.5 |
| COLL molecular collisions | PET-256 gives force MAE $23.1\ \mathrm{meV\,\mathring A^{-1}}$ and energy MAE $12.0\ \mathrm{meV/molecule}$; ECSE changes the latter to 11.9 with the same reported force MAE. | Appendix C.1, Table 2, p. 20 |
| Random methane geometries | Energy learning curves improve strongly with training-set size, for both energy-only and energy–force fitting. These are isolated distorted methane molecules. | Figure 3, p. 9; Appendix C.6 |
| High-entropy alloys | Strong hold-out errors do not ensure extrapolation: at 5000 K, PET's energy MAE is $152\ \mathrm{meV/atom}$ versus 48 for HEA25-4-NN. | Appendix C.4, pp. 21–22 |
| HME21 | A single PET model is slightly worse than MACE on the reported force-vector error; a five-model ECSE ensemble improves that metric to $128.5\ \mathrm{meV\,\mathring A^{-1}}$. | Appendix C.2, Table 3, p. 21 |
| QM9 and MnO | Vector dipoles demonstrate covariant outputs; MnO demonstrates extra spin inputs. PET is competitive but does not give the smallest QM9 atomization-energy error. | Section 7; Appendices C.3, C.7–C.8 |

The COLL energy unit above follows Appendix Table 2, which explicitly says
meV/molecule; the blanket energy-unit statement in the main Table 1 caption
is inconsistent with that detailed table. Force-component MAE and
force-vector-error MAE are also distinct metrics (Appendix Table 3).

These experiments establish architectural capability. They do not establish
MOF-5 adsorption accuracy, methane–framework force constants, or the accuracy
of a heat-capacity curve. In particular, the methane benchmark contains no
framework or host–guest interactions.

## Implementation in this MOF-5 repository

### Architecture, fitted potential, and execution backend

PET-MAD is a later general-purpose fit on a chemically and structurally diverse
DFT dataset. Its published model retains approximate rotational symmetry from
augmentation, rather than applying ECSE. Its reported rotational discrepancies
are usually smaller than its prediction errors; this does not validate a
particular MOF-5 checkpoint. ([PET-MAD paper](https://doi.org/10.1038/s41467-025-65662-7),
model architecture and benchmarking sections.)

The local PET-MAD and PET-SOL conversions currently record the same architecture
settings below. Matching architecture does not mean matching learned weights
or training data. These values were read from
`models/pet-mad-1.5-s_40nn_jax/metadata.yaml` and
`models/pet_sol-s-best_nostress_jax/metadata.yaml`; the generated model files
are intentionally excluded from version control.

| Metadata field | Local value | Meaning |
| --- | --- | --- |
| `cutoff` | $8.0\ \mathring A$ | Outer neighbor-search cutoff |
| `cutoff_width` | $0.5\ \mathring A$ | Cutoff switching width |
| `num_neighbors_adaptive` | 40 | Adaptive neighbor-selection setting |
| `num_gnn_layers` | 3 | Message-passing depth |
| `num_attention_layers` | 1 | Attention layers per message-passing block |
| `d_pet` | 256 | Token width |
| `num_heads` | 8 | Number of attention heads |
| `d_feedforward` | 512 | Transformer feed-forward width |
| `d_node`, `d_head` | 1024, 256 | Additional model/readout dimensions as named in the metadata |
| `num_species` | 102 | Species-embedding capacity; not a claim that all species are equally represented in training |

The original central-token and readout equations above explain the 2023
architecture; later checkpoint implementations can modify those details.
For example, inspection of the locally installed `metatrain` 2026.3.1
(`pet/modules/transformer.py` and `pet/model.py`) shows these concrete choices:

- `CartesianTransformer` linearly embeds $(x_{ij},y_{ij},z_{ij},r_{ij})$,
  adding the scalar distance to the three Cartesian components. The first
  block receives species embeddings as its initial messages; subsequent
  blocks also concatenate a neighbor-species embedding.
- The wider central features are contracted to the token width for attention
  and expanded again afterward. With the recorded dimensions, each of the
  eight attention heads has width $256/8=32$. The metadata's `d_head=256`
  is a **readout** width, not the width of an attention head.
- Padded neighbors are masked. Cutoff factors enter attention as a logarithmic
  additive mask, equivalent to multiplying the unnormalized softmax weights
  (with a small numerical floor). The implementation provides manual
  attention operations supporting second backpropagation.
- In the residual feature path, a reverse-neighbor index routes outgoing
  edge tokens to the receiving environments; incoming messages are updated
  by averaging the previous message and the reversed new output. Node and
  edge features from successive blocks are retained for readout.

These observations document the installed PyTorch implementation; the
checkpoint version and pinned PET-JAX conversion still determine compatibility.
The following project integration is directly visible in the repository:

1. [Model export](../mof_heat_capacity/models.py) uses `upet.save_upet` to
   produce the metatomic `.pt` model from the selected `.ckpt`. The
   [MD entry point](../mof_heat_capacity/simulation/md.py) uses the configured
   LAMMPS path for production classical NPT. Stress support and validation
   must be recorded for the selected export.
2. [Harmonic analysis](../mof_heat_capacity/analysis/harmonic.py) converts the
   matching checkpoint into PET-JAX parameter and metadata files; the
   [setup script](../scripts/setup/install_sadmof.sh) pins PET-JAX to
   `ed0add4`. Conversion transfers the fitted model rather than retraining it.
3. SADMOF constructs a scalar energy callable and obtains Hessian–vector
   products by forward-over-reverse AD. ASDEX coloring combines derivative
   directions and reconstructs a sparse Hessian. Fixed-cell optimized minima
   are the production reference structures.
4. The [hybrid template](../configs/mof5_100ch4_hybrid_npt.toml) currently uses
   `float32`, `hops=3`, `chunk_size=1`, `remat=true`, and `shadow=false`.
   These are analysis settings, distinct from the learned network parameters.

Two implementation qualifications matter for interpreting the report's
statement that MD and Hessians use the same PES. First, the sparse Hessian
call explicitly sets `no_shadow=True`: the SADMOF wrapper stops derivatives
through the adaptive cutoff. It therefore follows a specified derivative
convention rather than unrestricted differentiation of every selection
operation. Agreement with the MD force convention must be checked; identical
weights alone do not demonstrate it. Second, a three-block network does not
imply an exact three-hop Hessian pattern. SADMOF's local
`external/sadmof/src/sadmof/sparse/pattern.py` identifies seven hops as
structurally exact for its three-layer PET-MAD-S graph construction; the
project's three-hop setting is a truncation requiring convergence checks.
See [the Hessian implementation notes](SADMOF_HESSIANS.md) for both details.

### Consequences for the heat-capacity calculation

For fixed composition, species baselines have zero Cartesian derivatives and
do not change forces or Hessians. The learned energy scale does affect the
derivatives and must be preserved during conversion. PET supplies the PES;
the quantum statistics enter later through normal-mode heat capacities.

The relevant validation follows from the architecture and the project code:

- Compare energies and forces between the metatomic and PET-JAX paths, with
  consistent derivative conventions and selected model files.
- Rotate positions **and the periodic cell** together to test energy symmetry,
  force covariance, and stability of the vibrational spectrum.
- Check forces, temperatures, and framework/guest stability in trajectories;
  check Hessian precision, sparsity depth, cutoff behavior, and low-frequency
  modes at optimized structures.
- Validate the model for Zn–O nodes, linkers, methane, and their interactions.
  Architectural expressiveness and unrelated benchmark errors do not quantify
  the resulting heat-capacity uncertainty.

ECSE is explained here as a contribution of the paper. It is not enabled by
the project workflow, and adding it would change both model evaluation and
derivatives. The current method remains classical loaded-system NPT plus a
harmonic quantum correction from matching optimized loaded structures, with
a separately relaxed empty-framework Hessian as reference.
