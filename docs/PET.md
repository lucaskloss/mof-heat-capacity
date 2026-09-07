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

The displayed equations below serve two purposes. Equations tied to a paper
section or equation number restate the authors' construction. The additional,
unnumbered equations are supporting derivations that make standard background
explicit; they are not quotations or extra claims attributed to the paper.

## Scientific motivation

An MLIP replaces repeated electronic-structure calculations by a learned map
from atomic species, positions, and the periodic cell to potential energy.
An extensive energy can be written as a sum of learned local contributions:

$$U_\theta(\mathbf R,\mathbf h,\mathbf Z)=\sum_i b_{Z_i}+\sum_i\varepsilon_i(\mathcal A_i;\theta).$$

The species-dependent baselines $b_Z$ are constant at fixed composition.
Forces follow from $\mathbf F_i=-\partial U_\theta/\partial\mathbf R_i$.
Differentiating this same scalar energy gives conservative forces, but does
not by itself guarantee rotational symmetry or accuracy.

More explicitly, let $Q\in SO(3)$ be a proper rotation, $\mathbf a$ a
translation, and $\pi$ a permutation of atoms that also permutes their
species. The desired scalar-energy symmetries are

$$U_\theta(Q\mathbf R+\mathbf a,Q\mathbf h,\mathbf Z)=U_\theta(\mathbf R,\mathbf h,\mathbf Z),\qquad U_\theta(\pi\mathbf R,\mathbf h,\pi\mathbf Z)=U_\theta(\mathbf R,\mathbf h,\mathbf Z).$$

Here $Q\mathbf R+\mathbf a$ means applying the same transformation to every
atom, and rotating the cell with the atoms is essential for a periodic
system. If the first equality holds exactly, differentiating it gives force
covariance and the corresponding transformation of each Hessian block:

$$\mathbf F_i(Q\mathbf R,Q\mathbf h)=Q\mathbf F_i(\mathbf R,\mathbf h),\qquad H'_{i\alpha,j\beta}=\sum_{\gamma\delta}Q_{\alpha\gamma}H_{i\gamma,j\delta}Q_{\beta\delta}.$$

Energy-derived forces are conservative because, wherever $U_\theta$ is twice
differentiable, their cross derivatives obey

$$\frac{\partial F_{i\alpha}}{\partial R_{j\beta}}=\frac{\partial F_{j\beta}}{\partial R_{i\alpha}}=-H_{i\alpha,j\beta}.$$

Thus energy conservation, covariance, and physical accuracy are separate
properties: differentiability supplies the first relation, symmetry of the
energy supplies the second, and neither alone guarantees agreement with
reference electronic-structure data.

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

The periodic image vector $\mathbf n\in\mathbb Z^3$ is part of the neighbor
identity. Reversing a directed edge also reverses the relevant image,
$\mathbf r_{ji,-\mathbf n}=-\mathbf r_{ij\mathbf n}$, but PET is free to
assign different learned messages to the two directions. Smooth energy and
force removal requires at least $f_c(R_c)=f_c'(R_c)=0$; making the second
derivative continuous is also desirable when computing Hessians. A common
stronger convention with a unit inner region is

$$f_c(r)=1\ \text{in the inner region},\qquad f_c(R_c)=f_c'(R_c)=f_c''(R_c)=0.$$

For intuition, one possible switching function between an inner radius $R_s$
and the outer cutoff is a quintic taper, with

$$s=\frac{r-R_s}{R_c-R_s},\qquad f_c(r)=\begin{cases}1,&r\le R_s,\\1-10s^3+15s^4-6s^5,&R_s<r<R_c,\\0,&r\ge R_c.\end{cases}$$

This last expression illustrates the smoothness conditions; it does not
assert that every PET implementation uses this particular analytic taper.

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

$$Q_h=XW_h^Q,\qquad K_h=XW_h^K,\qquad V_h=XW_h^V,$$

$$\alpha_{ab}=\frac{\exp(\mathbf q_a^\mathsf T\mathbf k_b/\sqrt{d_k})\,c_b}{\sum_c\exp(\mathbf q_a^\mathsf T\mathbf k_c/\sqrt{d_k})\,c_c},\qquad \mathbf y_a=\sum_b\alpha_{ab}\mathbf v_b.$$

For $H$ heads, the per-head outputs are concatenated and projected back to the
token width,

$$\operatorname{MHA}(X)=\operatorname{Concat}(Y_1,\ldots,Y_H)W^O,$$

and a pre-normalization transformer layer can be summarized as

$$X'=X+\operatorname{MHA}(\operatorname{LN}X),\qquad X''=X'+\operatorname{FFN}(\operatorname{LN}X').$$

These equations expose the roles of the learned projections, parallel heads,
normalization, and residual paths; exact ordering is implementation dependent.

Here $c_0=1$ for the central token and $c_j=f_c(r_{ij})$ for a neighbor.
The cutoff multiplies the attention weights **before renormalization**, so a
departing neighbor also disappears smoothly from the normalization. Multiple
heads, feed-forward transformations, and residual connections provide the
transformer's flexibility. There is no positional encoding based on arbitrary
neighbor ordering: permuting neighbors permutes their output tokens.

Algebraically, if $P$ permutes the token rows, then the shared projections
give $Q(PX)=PQ(X)$ and similarly for $K$ and $V$. Provided the cutoff weights
and padding mask are permuted with their tokens, row-wise softmax and the
weighted sum consequently satisfy

$$\operatorname{Attention}(PX)=P\operatorname{Attention}(X).$$

The operation is therefore permutation **equivariant** at token level. A sum,
average, or designated central-token readout then makes the scalar energy
permutation **invariant**.

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

The angular information available to attention can be seen from the elementary
identity

$$\mathbf r_{ij}^{\mathsf T}\mathbf r_{ik}=r_{ij}r_{ik}\cos\theta_{jik}.$$

A nonlinear network acting jointly on Cartesian neighbor vectors can thus
construct angle-dependent and higher-body functions without receiving
$\theta_{jik}$ as a hand-designed feature. After $L$ message-passing blocks,
information can in principle travel along graph paths of length up to $L$;
this receptive-field statement concerns forward features and is not, by
itself, an exact bound on the Hessian sparsity pattern.

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

Although the readout is written as a sum, differentiation couples all atoms
that influenced a contribution. If $u_a$ denotes any central or edge readout,

$$\mathbf F_k=-\sum_a\frac{\partial u_a}{\partial\mathbf R_k},\qquad H_{k\alpha,l\beta}=\sum_a\frac{\partial^2u_a}{\partial R_{k\alpha}\partial R_{l\beta}}.$$

Consequently, an edge readout is not restricted to a radial pair force, and a
local forward model can have Hessian couplings extending beyond one neighbor
shell through its message-passing dependencies.

## Symmetry and the separate ECSE construction

Relative coordinates give translation invariance, and shared operations plus
symmetric readout give permutation invariance. Smooth activations and cutoff
handling give smooth geometry dependence. The unconstrained Cartesian
backbone does **not** enforce exact rotational invariance. Randomly rotating
training structures encourages approximate invariance but does not prove it.

ECSE constructs local coordinate frames from ordered pairs of noncollinear
neighbors, evaluates the backbone in those frames, and combines its outputs.
One explicit right-handed frame construction is

$$\mathbf e_1=\widehat{\mathbf r}_{ij},\qquad \mathbf e_2=\frac{\widehat{\mathbf r}_{ik}-(\widehat{\mathbf r}_{ik}\!\cdot\!\mathbf e_1)\mathbf e_1}{\left\lVert\widehat{\mathbf r}_{ik}-(\widehat{\mathbf r}_{ik}\!\cdot\!\mathbf e_1)\mathbf e_1\right\rVert},\qquad \mathbf e_3=\mathbf e_1\times\mathbf e_2,\qquad Q_{ijk}=[\mathbf e_1\ \mathbf e_2\ \mathbf e_3].$$

The denominator explains why collinear neighbor pairs must receive zero
weight or special handling. For a scalar local energy,

$$\varepsilon_i^{\mathrm{ECSE}}=\frac{\sum_{j\ne k}w_{ijk}\,\varepsilon_i^0(Q_{ijk}^{\mathsf T}\mathcal A_i)}{\sum_{j\ne k}w_{ijk}}.$$

The frame matrix $Q_{ijk}$ rotates with the environment, so coordinates
expressed in that frame are unchanged by an overall rotation. For vector or
tensor targets, the predicted components must be transformed back to the
original frame before averaging. The construction in Appendix F.1 uses proper
rotations, $SO(3)$; reflection invariance is an additional requirement and
should not be inferred solely from this formula.

Indeed, under a global rotation $S$, the constructed frame becomes
$Q'_{ijk}=SQ_{ijk}$, and therefore

$${Q'}_{ijk}^{\mathsf T}(S\mathbf r)=Q_{ijk}^{\mathsf T}S^{\mathsf T}S\mathbf r=Q_{ijk}^{\mathsf T}\mathbf r.$$

This identity is the core of exact scalar invariance. If the unconstrained
backbone instead predicts a vector $\mathbf v^0$ or a rank-two tensor $T^0$
in local-frame components, their laboratory-frame forms are

$$\mathbf v_{ijk}=Q_{ijk}\mathbf v^0_{ijk},\qquad T_{ijk}=Q_{ijk}T^0_{ijk}Q_{ijk}^{\mathsf T},$$

which transform as $\mathbf v'_{ijk}=S\mathbf v_{ijk}$ and
$T'_{ijk}=ST_{ijk}S^{\mathsf T}$ before ensemble averaging.

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

Writing a frame pair as a single index $a$, define
$p_a=w_a/\sum_b w_b$ and $\bar\varepsilon=\sum_a p_a\varepsilon_a$.
For positive active weights and any coordinate $q$, direct differentiation
gives

$$\frac{\partial\bar\varepsilon}{\partial q}=\sum_a p_a\frac{\partial\varepsilon_a}{\partial q}+\sum_a p_a(\varepsilon_a-\bar\varepsilon)\frac{\partial\log w_a}{\partial q}.$$

The first term contains the derivative of the backbone input, including the
coordinate-dependent frame $Q_a$. The second is the response of the normalized
frame weights. It vanishes only in special cases, so omitting it generally
does not produce the gradient of the ECSE energy.

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
validation-set mean squared errors. The species baselines can be understood as
a linear least-squares problem over structures $s$,

$$\mathbf b^*=\underset{\mathbf b}{\operatorname{argmin}}\sum_s\left(E_s^{\mathrm{ref}}-\sum_Z n_{sZ}b_Z\right)^2,$$

where $n_{sZ}$ counts atoms of species $Z$. Training the network on the
residual energy $E_s^{\mathrm{ref}}-\sum_Zn_{sZ}b_Z$ removes a large,
composition-dependent offset. Since these baselines are coordinate
independent, $\partial b_Z/\partial\mathbf R_i=0$: restoring them changes
reported total energies but not forces or Hessians.

A schematic normalized joint objective is

$$\mathcal L=\lambda_E\frac{\operatorname{MSE}(U_\theta,U^{\mathrm{ref}})}{\overline{\operatorname{MSE}}_{E,\mathrm{val}}}+\lambda_F\frac{\operatorname{MSE}(-\nabla_{\mathbf R}U_\theta,\mathbf F^{\mathrm{ref}})}{\overline{\operatorname{MSE}}_{F,\mathrm{val}}},$$

where the barred denominators denote the moving validation-error scales. This
form shows why energy and force terms with different units and magnitudes can
both influence optimization; their precise weights and averaging conventions
remain training choices. Rotational augmentation replaces a sample by
$(Q\mathbf R,Q\mathbf F)$ and encourages, but does not algebraically enforce,
$\mathbf F_\theta(Q\mathbf R)=Q\mathbf F_\theta(\mathbf R)$.

Most experiments use token width 128,
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

At an optimized fixed-cell structure $\mathbf R_0$, the local harmonic
expansion of that PES is

$$U(\mathbf R_0+\Delta\mathbf R)\approx U(\mathbf R_0)+\frac12\Delta\mathbf R^{\mathsf T}H\Delta\mathbf R,\qquad H=\left.\nabla_{\mathbf R}^{2}U\right|_{\mathbf R_0},$$

because $\nabla_{\mathbf R}U(\mathbf R_0)\approx0$. Automatic differentiation
can evaluate a Hessian–vector product without first materializing every entry,

$$H\mathbf v=\left.\frac{\mathrm d}{\mathrm d\epsilon}\nabla_{\mathbf R}U(\mathbf R+\epsilon\mathbf v)\right|_{\epsilon=0}.$$

After reconstruction, mass weighting and diagonalization convert Cartesian
curvatures into normal-mode frequencies:

$$D_{i\alpha,j\beta}=\frac{H_{i\alpha,j\beta}}{\sqrt{m_i m_j}},\qquad D\mathbf e_k=\omega_k^2\mathbf e_k.$$

For every retained positive-frequency mode, define
$x_k=\hbar\omega_k/(k_BT)$. Its quantum and classical harmonic heat capacities
are

$$C_{V,k}^{\mathrm{qn}}=k_B\frac{x_k^2e^{x_k}}{(e^{x_k}-1)^2},\qquad C_{V,k}^{\mathrm{cl}}=k_B,$$

so $C_{V,k}^{\mathrm{qn}}\to k_B$ for $x_k\to0$ and
$C_{V,k}^{\mathrm{qn}}\to0$ for $x_k\to\infty$. This makes the harmonic
quantum-minus-classical correction

$$\Delta C_V^{\mathrm{har}}(T)=\sum_{k\in\mathcal M}\left[C_{V,k}^{\mathrm{qn}}(T)-k_B\right],$$

where $\mathcal M$ is one consistently validated mode set. The project adds
this correction from a loaded-system Hessian to the loaded classical-MD
result. In the current NPT route, the conceptual combination is

$$C_P^{\mathrm{approx}}(T)=\frac{\mathrm d}{\mathrm dT}\left\langle U+K+P_{\mathrm{ext}}V\right\rangle_{NPT}+\Delta C_V^{\mathrm{har}}(T).$$

The first term retains classical anharmonic motion sampled by MD; the
quantum-minus-classical correction replaces the classical harmonic statistics
of the same loaded modes by their quantum statistics. The use of a fixed-cell
$C_V$ correction with an NPT $C_P$ derivative is an explicit approximation,
not a consequence of PET.

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
