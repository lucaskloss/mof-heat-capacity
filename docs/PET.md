# Point Edge Transformer: a step-by-step reading of the paper

These notes follow Sergey N. Pozdnyakov and Michele Ceriotti, *Smooth, exact
rotational symmetrization for deep learning on point clouds*, NeurIPS 2023.
The source is the supplied [PET.pdf](PET.pdf), arXiv v3, 6 February 2024.
All paper equation and page numbers below refer to that PDF; an
[online copy](https://arxiv.org/abs/2305.19302v3) is also available.

The paper presents two contributions. **PET** is a neural network that maps
atomic geometry and species to predictions. **ECSE** is a separate procedure
that makes a suitable network exactly equivariant under proper rotations.
Understanding PET does not require implementing ECSE first. PET-MAD is a later
trained potential, not the name of the architecture or a checkpoint supplied
with this 2023 paper.

We begin with one PET forward evaluation, then explain training and ECSE.
Equations identified by a paper number use its notation. Extra equations that
unpack prose or connect the model to this project are labelled as explanatory
or assembled expressions. Architectural motivations and design implications
are explanations of the construction, not additional claims of the paper.

## Notation and the overall computation

| Paper symbol | Meaning |
| --- | --- |
| $A_i$ | Environment centered on atom $i$, with neighbors inside $R_c$. |
| $s_i$ | Chemical species of atom $i$; called $Z_i$ elsewhere in the project. |
| $\mathbf r_{ij}$, $r_{ij}$ | Displacement from $i$ to $j$, and its length. |
| $k=1,\ldots,n_{\mathrm{GNN}}$ | Message-passing block index. |
| $n_{\mathrm{TL}}$ | Number of transformer layers inside each block. |
| $d_{\mathrm{PET}}$ | Number of components in a token or edge message. |
| $E_c^{(k)}$, $E_n^{(k)}$, $E_r^{(k)}$ | Central-species, neighbor-species, and position embedding maps. |
| $L^{(k)}$, $M^{(k)}$ | Cartesian embedding layer and token-building MLP. |
| $\tilde x_{ij}^{(k-1)}$ | Incoming message from $j$ to central atom $i$. |
| $t_j^{(k)}$ | Input token for neighbor $j$ in the current $i$-centered environment. |
| $x_i^{(k)}$, $x_{ji}^{(k)}$ | Transformer output for the center and for neighbor $j$, respectively. |
| $H_c^{(k)}$, $H_n^{(k)}$ | Learned maps from central and neighbor output tokens to target contributions. |
| $y_i^{\mathrm{bond}\,(k)}$ | Combined edge contribution for center $i$ at block $k$. |
| $\tilde E$, $E$, $F_{i\alpha}$ | Predicted energy, reference energy, and reference force component. |

The paper writes $t$ and $x$ without boldface even though they are feature
vectors. The tilde on $\tilde x$ distinguishes the incoming message used in
Eq. (5) from the output token $x$. The central atom is implicit in $t_j^{(k)}$:
the same atom $j$ has different tokens in different environments.

The index order deserves attention: $\tilde x_{ij}$ arrives **at $i$ from
$j$**, whereas $x_{ji}$ is produced **at $i$ for $j$**. Do not silently replace
both by an undirected pair feature. The directions can carry different data.

All centers are processed with shared parameters at a given block. Blocks have
their own learned maps, indicated by the superscript $(k)$. The figure is a
schematic assembled from Section 6, Figure 2, and Appendices A–B.

## 1. Build local environments and retain directions

**Source: Section 2, p. 2; Section 6 and Figure 2, pp. 6–8.**

For every central atom $i$, collect its neighbors inside $R_c$. Each neighbor
supplies its species $s_j$ and the three Cartesian components of
$\mathbf r_{ij}$. For an isolated structure, the displacement is
$\mathbf r_j-\mathbf r_i$.

For this project's periodic structures, the corresponding bookkeeping is
(explanatory extension, with cell vectors in the columns of $\mathbf h$)

$$\mathbf r_{ij}=\mathbf r_j\mathbf -\mathbf r_i,\qquad \mathbf \qquad r_{ij}<R_c.$$

Each image is a separate neighbor entry; exclude the center's zero-displacement
self entry. Below, $j$ abbreviates a neighbor entry, with its image label
understood. The reverse edge carries the opposite displacement and image
shift.

**Why this step matters.** Relative positions remove dependence on the choice
of origin. A cutoff bounds the amount of work for each center. Cartesian
vectors retain directional information that distances to the center alone
would discard: two neighbors can have the same distances but different angles.
PET lets the network learn how that information matters. Its hidden features
are not required to transform as physical vectors under rotations.

An edge here means a neighbor-list connection, not necessarily a chemical
bond. A single block sees one cutoff neighborhood. Later blocks receive
messages containing information from other neighborhoods, so the final
receptive field is larger than $R_c$.

## 2. Embed species and Cartesian geometry

**Source: Appendix A, p. 18, immediately before Eq. (5).**

An embedding converts an input into a vector of learned features. For chemical
species, $E_c^{(k)}$ and $E_n^{(k)}$ act as separate learned lookup tables:

$$E_c^{(k)}(s_i)\in\mathbb R^{d_{\mathrm{PET}}},\qquad E_n^{(k)}(s_j)\in\mathbb R^{d_{\mathrm{PET}}}.$$

These dimensions make explicit what the appendix describes. Separate tables
allow the same species to have different representations in the central and
neighbor roles. Species labels are categories; their numerical identifiers
are not treated as a continuous chemical coordinate.

The paper's position embedding is

$$E_r^{(k)}(\mathbf r_{ij})=\sigma\!\left(L^{(k)}(\mathbf r_{ij})\right),\qquad L^{(k)}:\mathbb R^3\longrightarrow\mathbb R^{d_{\mathrm{PET}}}.$$

Here $L^{(k)}$ is the learned linear layer and $\sigma$ is SiLU, applied to each
component. Explicitly, $\operatorname{SiLU}(z)=z/(1+e^{-z})$. The embedding
maps geometry to the same feature width as the species and message inputs;
its nonlinearity helps build flexible functions of geometry. SiLU is smooth,
which matters when differentiating the eventual energy.

The paper denotes this map by $E_r^{(k)}$; there is no separate $p_{ij}$ symbol.
This Cartesian position embedding is also different from a sequence-position
embedding in a language model: arbitrary neighbor-list order has no physical
meaning.

## 3. Construct the transformer tokens

**Source: Appendix A, Eq. (5), p. 18.**

For blocks after the first, neighbor $j$ provides three feature vectors: its
incoming message, its species embedding, and its position embedding. The
paper combines them as

$$t_j^{(k)}=M^{(k)}\!\left[\operatorname{concatenate}\!\left(\tilde x_{ij}^{(k-1)},E_n^{(k)}(s_j),E_r^{(k)}(\mathbf r_{ij})\right)\right].$$

Concatenation places the three vectors end to end. It produces
$3d_{\mathrm{PET}}$ components, keeping the three inputs separately accessible.
$M^{(k)}$ is an MLP with one hidden layer that mixes them and returns
$d_{\mathrm{PET}}$ components. This restores the fixed width needed for the
transformer and the next round of messages.

**First block.** No earlier messages exist. The appendix omits that input, so
$M^{(1)}$ maps $2d_{\mathrm{PET}}$ to $d_{\mathrm{PET}}$. Written explicitly
from that description,

$$t_j^{(1)}=M^{(1)}\!\left[\operatorname{concatenate}\!\left(E_n^{(1)}(s_j),E_r^{(1)}(\mathbf r_{ij})\right)\right].$$

The central token is simply $E_c^{(k)}(s_i)$. It has no displacement input.
For $m$ neighbors, the transformer therefore receives $m+1$ tokens, each of
width $d_{\mathrm{PET}}$. The central token supplies the identity of the atom
whose environment is being processed and provides a place from which to read
an environment-level contribution after attention.

**Alternative in Appendix A.** One may initialize messages with neighbor
species embeddings and omit the separate species input in subsequent token
construction. Then every $M^{(k)}$ takes $2d_{\mathrm{PET}}$ inputs. This is an
alternative architecture setting, not an additional step to apply on top of
Eq. (5). The appendix also describes adding scalar atomic properties such as
collinear spins, changing the geometry/property input from three to four
components. Such inputs must be available when making predictions.

## 4. Mix tokens with smooth local attention

**Source: Section 6, pp. 7–8; Appendix F.1, Eq. (9), p. 24.**

The transformer allows every token to use information from the center and
other neighbors in the same environment. Its output for one neighbor can
therefore depend on the entire local geometry. It retains one output per
input token, so neighbor identities survive this mixing.

For orientation, ordinary scaled dot-product attention computes learned query,
key, and value vectors and uses

$$\alpha_{ab}=\frac{\exp(\mathbf q_a^{\mathsf T}\mathbf k_b/\sqrt{d_k})}{\sum_c\exp(\mathbf q_a^{\mathsf T}\mathbf k_c/\sqrt{d_k})}.$$

This is an explanatory transformer equation, not a numbered equation in the
PET paper. Here $a,b,c$ label tokens within one environment; $d_k$ is the
query/key dimension, unrelated to the message-block index $k$.
A query–key score determines how strongly token $a$ uses token $b$'s value.
Multiple attention heads learn different such combinations.

### Why the cutoff must enter the normalization

If a neighbor were abruptly removed at $R_c$, both the attention numerator
and denominator would change. Multiplying only the final energy contribution
by a cutoff would not undo that neighbor's influence on the other tokens.
Section 6 therefore multiplies each attention coefficient by the source
neighbor's cutoff and **renormalizes**. Using the explanatory shorthand
$c_b=f_c(r_{ib}\mid R_c,\Delta_{R_c})$ for neighbors and $c_b=1$ for the
central token, its rule is

$$\alpha_{ab}\ \leftarrow\ \frac{\alpha_{ab}c_b}{\sum_c\alpha_{ac}c_c},\qquad \text{attention output for token }a=\sum_b\alpha_{ab}\mathbf v_b.$$

As $c_b$ tends to zero, that neighbor disappears from both the numerator and
denominator. Its value can no longer affect the surviving tokens. The
central token has unit cutoff weight, keeping this denominator positive in
exact arithmetic. In padded implementations, nonexistent neighbors must
also receive zero weight; padding is not a physical atom.

### The cutoff actually given in the paper

Appendix F.1, Eq. (9), gives the following function. Define the scalar $x$
only inside the switching interval:

$$x=2\frac{r-R_c+0.5\Delta_{R_c}}{\Delta_{R_c}}.$$

$$f_c(r\mid R_c,\Delta_{R_c})=\begin{cases}1,&r\le R_c-\Delta_{R_c},\\\frac12\left[\tanh\!\left(\frac{1}{x+1}+\frac{1}{x-1}\right)+1\right],&R_c-\Delta_{R_c}<r<R_c,\\0,&r\ge R_c.\end{cases}$$

$\Delta_{R_c}>0$ is the width over which a neighbor is switched off. The
interior variable runs from $-1$ to $1$. Use the constant branches at the
endpoints rather than evaluating the singular fractions there. The resulting
function joins smoothly to its constant branches with vanishing derivatives.
A narrow interval still produces large derivatives inside it; mathematical
smoothness alone does not ensure well-behaved force constants.

### What repeats inside the transformer

The block contains $n_{\mathrm{TL}}$ transformer layers. Attention exchanges
information between tokens; token-wise feed-forward networks transform the
features; residual paths preserve information and help optimization. The
paper refers to a classical transformer rather than spelling out every
normalization and projection convention. A particular pre/post-normalization
formula should therefore not be attributed to this paper without checking
its released implementation.

The weights are shared across neighbor slots. Reordering neighbors merely
reorders their outputs: this is permutation equivariance. Summing suitable
outputs later produces a permutation-invariant energy. Attention nevertheless
contains weighted sums; the paper's description of “aggregation-free” message
passing means that messages are not all collapsed into one node vector before
being sent onward.

## 5. Read the outputs and pass messages to the next centers

**Source: Figure 2, p. 7, and Appendix A, p. 18.**

For the environment centered on $i$, the transformer returns the central
output $x_i^{(k)}$ and neighbor outputs $x_{ji}^{(k)}$. Each neighbor output
has two uses: predicting an edge contribution and constructing a message to
that neighbor. The central output supplies a central contribution.

Appendix A specifies a residual message update: output tokens are added to
previous messages. This message residual is distinct from the residual
connections inside each transformer layer. The updated message produced at
$i$ for $j$ must then be routed to the entry for sender $i$ in the environment
centered on $j$. In a periodic implementation, this requires matching the
reverse image edge as well as the atom indices. The paper describes the update
in prose; it does not give a numbered recurrence fixing all storage and
indexing conventions.

All environments at block $k$ use messages from block $k-1$; only after that
round are the new messages used by block $k+1$. This is a network evaluation,
not a sequence of molecular-dynamics time steps. Atomic positions remain
fixed throughout the forward pass.

For example, suppose $j$ neighbors both $i$ and $l$, but $l$ lies outside
$i$'s cutoff. At the first block, the environment of $j$ can encode information
about $l$ in the message sent to $i$. At the next block, $i$ can use that
information even though $l$ is absent from its own neighbor list. Increasing
$n_{\mathrm{GNN}}$ extends this communication range. Increasing
$n_{\mathrm{TL}}$ instead adds processing within each existing environment.
These are different ways of increasing model capacity, with different costs.

## 6. Convert features into energy contributions

**Source: Appendix A, Eqs. (6)–(7), p. 18; Figure 2, p. 7.**

A readout head is a learned map from a feature vector to the target, here a
scalar energy contribution. At each block the central contribution is
$H_c^{(k)}(x_i^{(k)})$. The summed edge contribution is exactly the form of
the paper's Eq. (7):

$$y_i^{\mathrm{bond}\,(k)}=\sum_{j\in A_i}H_n^{(k)}(x_{ji}^{(k)})f_c(r_{ij}\mid R_c,\Delta_{R_c}).$$

Cutoff weighting is needed again here. Attention suppresses a departing
neighbor's influence on the surviving tokens, but that neighbor's own output
token need not go to zero. Multiplying its readout removes its direct energy
contribution as it exits.

The paper also permits the weighted average, Eq. (6):

$$y_i^{\mathrm{bond}\,(k)}=\frac{\sum_{j\in A_i}H_n^{(k)}(x_{ji}^{(k)})f_c(r_{ij}\mid R_c,\Delta_{R_c})}{\sum_{j\in A_i}f_c(r_{ij}\mid R_c,\Delta_{R_c})}.$$

These are alternatives. The sum retains a direct dependence on the total
neighbor weight; the average normalizes it out. That changes the model and
must agree between training and inference. Unlike the attention denominator,
this average has no central token term. If every neighbor weight is zero,
Eq. (6) is undefined as written: a usable implementation needs a specified
convention, and a naive zero fallback does not by itself guarantee continuity.
The summed version naturally gives zero for an empty neighbor set.

Both central and edge predictions are collected across message-passing blocks.
An assembled bookkeeping expression for the network's energy contribution is

$$\tilde E_{\mathrm{network}}=\sum_{k=1}^{n_{\mathrm{GNN}}}\sum_i\left[H_c^{(k)}(x_i^{(k)})+y_i^{\mathrm{bond}\,(k)}\right].$$

The subscript “network” is introduced here to distinguish this sum from the
self-contributions added next; the paper does not print this combined formula
as a numbered equation. The network learns to distribute the target across
these terms. They are not independently measured atomic or bond energies.
In particular, an edge token depends on its environment and earlier messages,
so its readout is a many-body contribution. There is no physical pair-counting
argument for inserting an extra factor of one half into the paper's sum.

## 7. Restore the energy baseline and differentiate

**Source: Appendix B, “Self-contributions” and Eq. (8), p. 19.**

Before network training, the authors fit a linear model using species counts
and subtract that prediction from the reference energies. During inference,
they add it back. To make this prose explicit, introduce $b_s$ for the fitted
coefficient of species $s$ and write

$$E_{\mathrm{self}}=\sum_i b_{s_i},\qquad \tilde E=E_{\mathrm{self}}+\tilde E_{\mathrm{network}}.$$

These are explanatory symbols for Appendix B's atomic self-contributions.
They are the report's species energy baseline $U_{\mathrm{base}}$. This is a
constant at fixed composition, not a position-dependent reference force field.
Removing this simple composition dependence lets the neural network focus on
changes associated with geometry and interactions. The coefficients are fitted
on the training set and need not equal isolated-atom electronic energies.

Forces follow from the same predicted scalar energy, as used in Eq. (8):

$$\tilde F_{i\alpha}=-\frac{\partial\tilde E}{\partial r_{i\alpha}}.$$

Here $\alpha$ is a Cartesian component. Differentiation includes all paths by
which an atom influences any environment, token, message, cutoff, and readout.
It is not enough to differentiate only the contribution centered on that atom.
$E_{\mathrm{self}}$ has zero Cartesian derivatives. Defining forces as this
gradient makes them conservative where the energy is differentiable; it does
not independently establish rotational invariance or physical accuracy.

## 8. Train the model and choose its capacity

**Source: Appendix B, p. 19; Section 7, pp. 8–10; Appendix C, pp. 19–23.**

Training adjusts embeddings, transformer weights, token MLPs, and readout
heads together. For simultaneous energy and force training, the paper uses
Eq. (8):

$$L=w_E\frac{(\tilde E-E)^2}{\mathrm{MSE}_E}+\frac{\frac{1}{3N}\sum_{i\alpha}\left(-\frac{\partial\tilde E}{\partial r_{i\alpha}}-F_{i\alpha}\right)^2}{\mathrm{MSE}_F}.$$

$N$ is the number of atoms in this sample. $\mathrm{MSE}_E$ and
$\mathrm{MSE}_F$ are exponential moving averages of validation-set mean squared
errors, updated once per epoch. They set evolving scales for the two kinds of
residual; $w_E$ controls their relative importance. The loss above is written
for a configuration; training combines contributions from training samples.
Use consistent energy offsets for $\tilde E$ and $E$.

The factor $1/(3N)$ averages the force-component error. The numerator of the
energy term in Eq. (8) is the error in total energy, without an additional
per-atom division. Appendix C.2 discusses a different normalization for HME21;
it should not silently replace the general formula. Training on forces
requires differentiating the energy gradient with respect to model parameters.

The authors rotate configurations randomly at each training epoch. Force
labels must rotate with positions; periodic cells must rotate as well. This
teaches approximate rotational invariance but does not enforce an exact
identity for every possible input. Appendix B reports Adam optimization,
validation-based early stopping, and no dropout or weight decay for those
experiments. Its default architecture is $d_{\mathrm{PET}}=128$,
$n_{\mathrm{GNN}}=n_{\mathrm{TL}}=3$, four attention heads, and transformer
feed-forward width 512, with benchmark-specific exceptions.

### What to keep in mind when designing a PET model

| Choice | Purpose and consequence |
| --- | --- |
| $R_c$ | Sets the direct neighborhood and its information content. A larger radius increases neighbor count and attention cost. Finite-range message passing does not automatically supply correct long-range electrostatics. |
| $\Delta_{R_c}$ | Sets the switching width. Check derivatives through the entire model, including normalization and readout. |
| $d_{\mathrm{PET}}$ and head count | Control feature capacity and attention partitioning. Feature width is not a count of physical vector components or chemical species. |
| $n_{\mathrm{TL}}$ | Adds nonlinear processing of the same local token set without another inter-environment communication round. |
| $n_{\mathrm{GNN}}$ | Adds communication rounds, enlarging the receptive field and the cost of distributed evaluation. |
| Summed or averaged edge readout | Changes sensitivity to coordination and the handling required for empty neighborhoods. |
| Species-input variant | Determines whether the token MLP takes two or three feature groups after the first block. |
| Symmetry strategy | Choose augmentation, an ECSE construction, or an intrinsically equivariant model with their distinct training and inference requirements. |
| Training data and targets | Must cover the chemistry, structures, and derivatives needed in the intended simulation. Architecture alone cannot establish that coverage. |

For $m$ neighbors, each attention head forms an $(m+1)\times(m+1)$ matrix.
Thus its token-pair work is quadratic in local neighbor count, while total
work scales approximately linearly with atom count at bounded neighborhoods
and fixed architecture. This scaling argument is an explanation of the local
construction, not a full runtime model.

The water ablation in Section 7 compares the two depth parameters: with twelve
transformer layers in total, six message-passing blocks with two layers each
outperform twelve blocks with one layer each in that experiment. It illustrates
why local processing and communication depth should be tuned separately,
rather than establishing a universal best choice. Appendix C contains dataset
splits, learning-rate choices, and exceptional settings; Appendix D points to
released code, checkpoints, and configuration files needed for reproduction.

## 9. Understand ECSE as a separate symmetry construction

**Source: Section 5, pp. 4–6; Appendix F, pp. 23–30.**

PET's relative coordinates and shared token operations provide translation
and permutation symmetries. Its unconstrained Cartesian embeddings do not
ensure rotational invariance. ECSE addresses this by evaluating a backbone
$\mathbf y_0$ in coordinate systems attached to the structure and combining
those predictions into $\mathbf y_S$. For a scalar energy, invariant means
unchanged by rotation; for a vector output, equivariant means rotated with
the input.

### Construct a frame from two neighbors

For each ordered, noncollinear neighbor pair $j,j'$, Appendix F.1 defines
$\hat v_1=\mathbf r_{ij}/r_{ij}$ and
$\hat v_2=\mathbf r_{ij'}/r_{ij'}$, then constructs

$$\mathbf u_1=\hat v_1\times\hat v_2,\qquad \hat u_1=\frac{\mathbf u_1}{\lVert\mathbf u_1\rVert},\qquad \hat u_2=\hat v_1\times\hat u_1.$$

The frame axes are $(\hat v_1,\hat u_1,\hat u_2)$, in that order. This is the
paper's right-handed construction. $\hat R_{jj'}^{-1}[A_i]$ denotes the
geometry expressed in that frame. A rigid rotation moves the axes and atoms
together, leaving those internal coordinates unchanged. For message-passing
models, every coordinate on which the prediction depends must be treated
consistently, not only the immediate shell.

### Weight frames and average predictions

Equation (4) assigns a weight

$$w_{jj'}=f_c(r_{ij}\mid R_c,\Delta_{R_c})f_c(r_{ij'}\mid R_c,\Delta_{R_c})q_c\!\left(\lVert\hat r_{ij}\times\hat r_{ij'}\rVert^2\mid\omega,\Delta_\omega\right).$$

Here $\hat r_{ij}$ denotes the unit displacement. The argument of $q_c$ is
the squared sine of the angle between the two neighbors. The radial factors
remove frames as their defining neighbors leave the cutoff. The angular factor
suppresses nearly collinear pairs, whose frame axes would be unstable.
Appendix F.1, Eqs. (10)–(11), gives two alternatives, $q_c^1$ and $q_c^2$:
both are zero up to $\omega$; over width $\Delta_\omega$ they switch smoothly
to $1$ and to their argument $z$, respectively.

Equation (3) then gives

$$\mathbf y_S(A_i)=\frac{\sum_{jj'\in A_i}w_{jj'}\hat R_{jj'}\!\left[\mathbf y_0\!\left(\hat R_{jj'}^{-1}[A_i]\right)\right]}{\sum_{jj'\in A_i}w_{jj'}}.$$

The sum runs over ordered nondegenerate pairs with active weights. The outer
rotation returns vector or tensor predictions to the original frame; it has
no effect on a scalar energy. Swapping $j$ and $j'$ generally defines another
frame. A plain unweighted average would change abruptly when new frames
appear, so the weights and their normalization are essential.
This construction enforces proper rotations; it does not by itself impose
reflection symmetry.

### What the remaining appendix adds

| Appendix | Problem addressed and design lesson |
| --- | --- |
| F.2 | Assumes finitely many neighbors in finite regions and a positive minimum interatomic separation. |
| F.3 | If all frame weights vanish, Eq. (3) becomes $0/0$. One remedy smoothly blends in an internally equivariant auxiliary model. A second, more restricted construction adapts angular thresholds and treats the collinear limit; it is not a general replacement for the auxiliary route for vector inputs/outputs. |
| F.4 | Defines a smooth adaptive inner radius $R_{\mathrm{in}}(A_i)$ to select nearby usable frame pairs. Smooth extrema and angular weights replace a hard choice of the two closest neighbors. The frame-selection cutoff need not equal the backbone's interaction cutoff. |
| F.5 | Prunes low-weight frames smoothly, and relaxes pruning when all frame weights are small. A hard threshold would undermine smoothness. |
| F.6–F.7 | Tight frame selection saves evaluations but can create large derivatives. Forces include derivatives of the frames and weights, so accurate averaged energies need not imply accurate forces. |
| F.8–F.9 | Discusses extra fixed rotational augmentations and how ECSE can be combined with models containing long-range information or contributions. |
| F.10 | Explains why independently symmetrizing messages changes the training problem: messages need compatible transformation rules between sending and receiving frames. |
| F.11–F.12 | Describes the proof-of-principle implementation using a global pool of frames, and numerical smoothness checks. Its computational overhead is not an intrinsic cost estimate for every possible ECSE implementation. |

In particular, differentiating a symmetrized energy requires differentiating
$\hat R_{jj'}$ and $w_{jj'}$ as well as the backbone prediction. Simply averaging
the backbone's forces omits terms. Appendix F.10 also explains why layerwise
ECSE cannot generally be added unchanged after training with whole-structure
rotations: unconstrained messages may encode orientation information that
independently chosen frames would destroy.

Appendix G concerns making other point-cloud architectures smooth. Its useful
general lesson is to audit neighbor selection and pooling as well as activation
functions: hard nearest-neighbor replacement, maximum operations, or sampling
can introduce nonsmooth behavior. Those alternative architectures are not
extra stages of PET.

## 10. Connection to the loaded-MOF project

The original architecture above explains how PET computes a learned scalar
energy. The later PET-MAD/PET-SOL checkpoints, their conversion, and their
neighbor-selection conventions determine the actual function evaluated in this
project. Their details belong to the
[implementation notes](../scripts/properties/HESSIANS.md). ECSE is a separate
construction and is not part of the project's current evaluation path.

For the report, identify the learned potential $U$ with the model's total
predicted energy $\tilde E$. The project's Cartesian Hessian is the further
derivative (project connection, not a PET architecture equation)

$$H_{i\alpha,j\beta}=\frac{\partial^2\tilde E}{\partial r_{i\alpha}\partial r_{j\beta}}.$$

The baseline has no effect on this derivative. Cutoffs, message paths, learned
energy scales, and derivative conventions do. The harmonic calculation then
uses the mass-weighted Hessian to obtain frequencies; quantum mode heat
capacities and the hybrid NPT correction are subsequent physics, not operations
inside PET. See [SADMOF.md](SADMOF.md) and the harmonic sections of the
[report](report/report.tex).

Before interpreting loaded-MOF heat capacities, check the matching MD and
Hessian energy/force conventions, rotation behavior, optimized minima, and
Hessian convergence. The paper's water, alloy, collision, and molecular
benchmarks illustrate architectural performance; they do not establish the
accuracy of a particular checkpoint for framework–guest interactions.
