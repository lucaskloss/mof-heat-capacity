# SADMOF: sparse Hessians and harmonic heat capacity

Marcel F. Langer, Adrian Hill, and Michele Ceriotti, *Truncated automatic sparse
differentiation for machine learning interatomic potentials*,
[supplied preprint](../pdf/SADMOF.pdf). Section 4 describes the method; Appendices C,
D, and H derive the reach, block coloring, and truncation error.
Commands, library calls, parameters, archives, and debugging are documented
beside the scripts in [HESSIANS.md](../../../scripts/properties/HESSIANS.md).

This explanation continues from “The Hessian and HVPs” in the
[report appendix](../../report/report.tex). It assumes you understand that an
HVP gives a weighted sum of Hessian columns, and that coloring lets us combine
columns when we can separate their entries afterward. The remaining question
is: **how can we know which entries may be nonzero before computing the
Hessian?** The paper answers this using the model's graph of atoms.

## 1. What is a hop?

The potential receives atomic positions and builds a **neighbor graph**. Its
vertices are atoms, and an edge connects a pair selected as neighbors by the
model. Selection is based on geometric neighborhoods and the model's cutoff
procedure. An edge need not represent a chemical bond. For PET, use the graph
after its adaptive neighbor selection; the paper's sparse pipeline stops
derivatives through the adaptive cutoff.

One **hop** means following one edge. The hop distance between two atoms is
the smallest number of edges needed to get from one to the other. Consider
this illustrative graph, with no connections other than those drawn:

```text
atom 1 —— atom 2 —— atom 3 —— atom 4
```

| Pair | Shortest route | Hop distance |
| --- | --- | --- |
| 1 and 1 | Stay at atom 1 | 0 |
| 1 and 2 | 1 → 2 | 1 |
| 1 and 3 | 1 → 2 → 3 | 2 |
| 1 and 4 | 1 → 2 → 3 → 4 | 3 |

Consequently, the neighborhood within two hops of atom 1 contains atoms
1, 2, and 3. The neighborhood within two hops of atom 2 contains all four
atoms. The count always depends on the starting atom and the graph.

A hop is not a time step, an AD pass, or a fixed distance in angstroms. Two
edges can have different lengths. Also, if the model selected a direct edge
between atoms 1 and 3, their hop distance would become one, even if the route
through atom 2 still existed. In a periodic structure, neighbor connections
can cross the simulation-cell boundary.

## 2. The adjacency matrix A records one-hop connections

For $N$ atoms, the adjacency matrix $A$ is an $N\times N$ binary matrix:

$$A_{ij}=\begin{cases}1,&\text{if atoms }i\text{ and }j\text{ share an edge},\\0,&\text{otherwise}.\end{cases}$$

Use an undirected graph: $A_{ij}=A_{ji}$. For the four-atom chain,

$$A=\begin{pmatrix}0&1&0&0\\1&0&1&0\\0&1&0&1\\0&0&1&0\end{pmatrix}.$$

Row 1 says that atom 1 has only atom 2 as a direct neighbor. Row 2 says that
atom 2 has atoms 1 and 3 as direct neighbors. The diagonal is zero here because
we have not drawn self-edges; we include each atom's coupling to itself separately.

Matrix powers let us find longer connections. With ordinary arithmetic,

$$ (A^2)_{ij}=\sum_{\ell=1}^{N}A_{i\ell}A_{\ell j}. $$

For example, $(A^2)_{13}$ receives a contribution from $\ell=2$, because
$A_{12}A_{23}=1$. This represents the route $1\to2\to3$. Ordinary matrix
multiplication counts two-edge walks; we only need to know whether at least
one exists. The paper therefore uses **Boolean arithmetic**: multiply with
AND and add with OR. Under this convention,

$$ (A^2)_{ij}=\bigvee_{\ell=1}^{N}(A_{i\ell}\land A_{\ell j}). $$

Here $\land$ means “both connections exist,” and $\lor$ means “at least one
possibility exists.” The Boolean square of the chain's adjacency matrix is

$$A^2=\begin{pmatrix}1&0&1&0\\0&1&0&1\\1&0&1&0\\0&1&0&1\end{pmatrix}.$$

Its diagonal entries are one because a two-step walk can go out and back,
such as $1\to2\to1$. Thus $A^2$ describes walks of exactly two edges, not
exclusively pairs whose *shortest* distance is two. This is why we combine
several powers to obtain a neighborhood.

## 3. What is the P matrix?

The matrix $P^{(k)}$ records which atom pairs are connected by **at most $k$
hops**, including zero hops:

$$P^{(k)}=I\lor A\lor A^2\lor\cdots\lor A^k.$$

The parenthesized superscript labels the chosen hop limit; $P^{(2)}$ is not
the square of $P$. The paper calls this construction $A^{(K)}$ in Equation 2;
$P^{(k)}$ is the notation used in the report and here. Each matrix has one
row and one column per atom. Equivalently,

$$P^{(k)}_{ij}=\begin{cases}1,&d_G(i,j)\leq k,\\0,&d_G(i,j)>k,\end{cases}$$

where $d_G$ is the shortest hop distance, with infinite distance for
disconnected pairs. The identity matrix $I$ supplies the diagonal, since
an atom is zero hops from itself.

For the chain, the one-hop pattern is

$$P^{(1)}=I\lor A=\begin{pmatrix}1&1&0&0\\1&1&1&0\\0&1&1&1\\0&0&1&1\end{pmatrix}.$$

Including two-hop connections gives

$$P^{(2)}=I\lor A\lor A^2=\begin{pmatrix}1&1&1&0\\1&1&1&1\\1&1&1&1\\0&1&1&1\end{pmatrix}.$$

For example, $P^{(1)}_{13}=0$ but $P^{(2)}_{13}=1$, because reaching atom 3
from atom 1 takes two hops. The only zero pair left in $P^{(2)}$ is 1–4,
together with its symmetric entry 4–1. At three hops, $P^{(3)}$ is all ones,
because every pair in this particular graph is reachable within three steps.

**P is a map of permitted matrix positions, not a matrix of force constants.**
It contains no derivative values, masses, or coupling strengths. Increasing
$k$ adds permitted pairs, without calculating their Hessian entries. The
patterns are nested: a one in $P^{(k)}$ remains a one in $P^{(k+1)}$.

The interpretation as a Hessian pattern requires a sufficient hop limit:

- At the model's full Hessian reach $K$, $P^{(K)}_{ij}=0$ guarantees that
  the corresponding Hessian block is zero under the paper's assumptions.
- $P^{(K)}_{ij}=1$ means the block may be nonzero. It can still vanish at a
  particular geometry or for particular model weights.
- With a deliberately smaller limit $k<K$, $P^{(k)}_{ij}=0$ means we omit
  that pair from reconstruction; its actual Hessian block may be nonzero.

## 4. Why can atoms more than one hop apart have a Hessian coupling?

For $N$ atoms, the Cartesian force-constant matrix is

$$H_{i\alpha,j\beta}=\frac{\partial^2 E}{\partial R_{i\alpha}\,\partial R_{j\beta}}.$$

This measures how changing coordinate $R_{j\beta}$ changes the energy
gradient with respect to $R_{i\alpha}$; equivalently, it is minus the
corresponding force derivative. Write the total energy as a sum of terms:

$$E=\sum_t\varepsilon_t,\qquad H_{i\alpha,j\beta}=\sum_t\frac{\partial^2\varepsilon_t}{\partial R_{i\alpha}\,\partial R_{j\beta}}.$$

If a term does not depend on both coordinates, its mixed derivative is zero.
Therefore, atoms can couple when they influence the **same energy term**,
even without an edge directly joining them.

For a concrete example, give each atom of a three-atom chain one scalar
displacement $x_i$, and consider a term centered on atom 2:

$$\varepsilon_2=\frac12(x_1-2x_2+x_3)^2.$$

Both endpoints are one hop from the center, so this term uses only the
center's one-hop neighborhood. Nevertheless,

$$\frac{\partial\varepsilon_2}{\partial x_1}=x_1-2x_2+x_3,\qquad\frac{\partial^2\varepsilon_2}{\partial x_1\partial x_3}=1.$$

Atoms 1 and 3 are two hops apart and have a nonzero Hessian entry. This
illustrates why using just the direct-neighbor adjacency matrix would miss
couplings. Dependence on both atoms makes a mixed derivative possible,
although it does not guarantee one for every function: a separable term
$g(x_1)+h(x_3)$ would have zero mixed derivative.

### Following this dependence through PET's equations

The following uses the original PET architecture in Section 6 and Appendix A
of the [PET paper](../pdf/PET.pdf); see also the longer [PET notes](PET.md).
The equations defining the neighborhood and grouping the transformer outputs
below are explanatory notation. The token equation is the paper's Equation 5,
and the summed edge readout follows its Equation 7.

For the original fixed-cutoff construction, define the neighbor indices and
the environment by

$$\mathcal N_i=\{j\ne i:\|\mathbf R_j-\mathbf R_i\|<R_c\},\qquad A_i=\{(\mathbf r_{ij},s_j):j\in\mathcal N_i\},\qquad\mathbf r_{ij}=\mathbf R_j-\mathbf R_i.$$

Here $A_i$ is an atomic environment, distinct from the adjacency matrix $A$.
For periodic systems, relative vectors include the appropriate cell-image
shifts. Modern PET checkpoints can adapt the cutoff selection; the argument
below uses the selected graph at a given geometry, with the sparse pipeline's
derivative convention described above.

Take a chain with selected edges 1–2 and 2–3, but no 1–3 edge. Then atom 1
has neighbor 2, and atom 2 has neighbors 1 and 3. A one-hop environment around
atom 2 therefore contains both endpoints.

PET's first block embeds each neighbor using its species and relative vector:

$$t_j^{(1)}=M^{(1)}\!\left[\operatorname{concatenate}\!\left(E_n^{(1)}(s_j),E_r^{(1)}(\mathbf r_{ij})\right)\right].$$

Within the environment centered at 2, the two neighbor tokens thus depend on
$\mathbf r_{21}$ and $\mathbf r_{23}$, respectively. Write the transformer
as $\mathcal T^{(1)}$ for convenience:

$$\left(x_2^{(1)},x_{12}^{(1)},x_{32}^{(1)}\right)=\mathcal T^{(1)}\!\left(E_c^{(1)}(s_2),t_1^{(1)},t_3^{(1)}\right).$$

Self-attention mixes tokens within this environment. Consequently, even the
output associated with neighbor 1 can depend on the input describing neighbor
3. The neighbor output is not a function of that pair's distance alone.

Using PET's summed readout, the energy contribution from this environment
and block is

$$\varepsilon_2^{(1)}=H_c^{(1)}(x_2^{(1)})+\sum_{j\in\{1,3\}}H_n^{(1)}(x_{j2}^{(1)})f_c(r_{2j}).$$

The symbols $H_c,H_n$ here are neural-network readout heads, not Hessian
matrices; the cutoff parameters have been suppressed for readability. At
fixed species, the entire expression can be collected into a learned
function $F_2$:

$$\varepsilon_2^{(1)}=F_2(\mathbf r_{21},\mathbf r_{23}).$$

This is an explanatory rewriting of PET's computation, not an assumed
quadratic energy. Applying the chain rule at a fixed selected graph gives

$$\frac{\partial^2\varepsilon_2^{(1)}}{\partial R_{1\alpha}\partial R_{3\beta}}=\frac{\partial^2F_2}{\partial r_{21,\alpha}\partial r_{23,\beta}}.$$

The right side can be nonzero because the nonlinear transformer mixes both
inputs. It contributes to the full Hessian block $H_{1\alpha,3\beta}$.
**A single environment already permits a two-hop coupling between its
neighbors.** This does not require atom 3 to be in atom 1's neighbor list.

Message passing expands the dependence further. For later blocks, PET's
Equation 5 includes the incoming message:

$$t_j^{(\ell)}=M^{(\ell)}\!\left[\operatorname{concatenate}\!\left(\tilde x_{ij}^{(\ell-1)},E_n^{(\ell)}(s_j),E_r^{(\ell)}(\mathbf r_{ij})\right)\right].$$

The block index is written $\ell$ to keep it distinct from the truncation
limit $k$. The message $\tilde x_{12}^{(1)}$ arrives at atom 1 from atom 2.
It is built from outputs of atom 2's first block, which can depend on atom
3. Hence the token for neighbor 2 in atom 1's second block is

$$\left.t_2^{(2)}\right|_{i=1}=M^{(2)}\!\left[\operatorname{concatenate}\!\left(\tilde x_{12}^{(1)},E_n^{(2)}(s_2),E_r^{(2)}(\mathbf r_{12})\right)\right].$$

The dependence follows the route

```text
position of atom 3 → environment centered at 2 → message from 2 to 1
                  → token at center 1 in the next block → energy
```

Atom 3 can therefore influence predictions at center 1 while remaining two
hops away on the unchanged neighbor graph. More blocks repeat this exchange.
Additional transformer layers inside one block mix the existing environment's
tokens; exchanging messages between environments extends the graph reach.

### From message-passing depth to the maximum Hessian reach

In **message passing**, each layer updates an atom's representation using
its neighbors' representations. After one round, information can have come
from one hop away; after two rounds, from two hops away, and so on. The set
of atoms that can influence an output is its *receptive field*.

For a **node readout**, an energy term is predicted at a central atom after
$L$ rounds. An atom $i$ may lie $L$ hops to one side of that center and an
atom $j$ another $L$ hops to the other side. Their separation can reach
$L+L$, so the maximum Hessian reach is $K=2L$.

For the **PET edge readout** analyzed in the paper, an energy term draws
on neighborhoods around two adjacent endpoints $u,v$. The longest route
between two atoms influencing that term can run

```text
atom i —— up to L edges —— u —— v —— up to L edges —— atom j
```

Joining these routes gives the bound $K=L+1+L=2L+1$. In a graph with
shortcuts the shortest route can be smaller; $K$ is the architectural
maximum. These bounds assume the architecture and derivative convention
studied in the paper.

| Model in the paper | $L$ | Exact reach $K$ |
| --- | --- | --- |
| MACE-MP-0 medium | 2 | 4 |
| PET-MAD-1.5 XS | 2 | 5 |
| PET-MAD-1.5 S | 3 | 7 |

Distinguish the three counts: $L$ describes the model's message-passing
depth, $K$ is its full Hessian reach, and $k$ is the hop limit chosen for
reconstruction. Setting `hops=3` selects $k=3$; it does not change the
number of layers or shorten the model calculation inside an HVP.

## 5. From atom pairs to Cartesian matrix entries

The pattern $P^{(k)}$ has shape $N\times N$, whereas the physical Hessian
has shape $3N\times3N$. Each atom has three coordinates, so one atom pair
corresponds to a block of nine possible derivatives:

$$H_{ij}=\begin{pmatrix}H_{ix,jx}&H_{ix,jy}&H_{ix,jz}\\H_{iy,jx}&H_{iy,jy}&H_{iy,jz}\\H_{iz,jx}&H_{iz,jy}&H_{iz,jz}\end{pmatrix}.$$

Replace each one in $P^{(k)}$ with a $3\times3$ block of ones and each zero
with a $3\times3$ block of zeros. Call this Cartesian binary mask $M_k$:

$$M_k=P^{(k)}\otimes\mathbf 1_{3\times3}.$$

The Kronecker product $\otimes$ means exactly this block replacement. For
the four-atom chain, $P^{(2)}$ is $4\times4$ and $M_2$ is $12\times12$.
Its zero 1–4 atom pair becomes nine zero positions in the corresponding
Cartesian block. A diagonal one includes the atom's full on-site block,
including mixed derivatives such as $H_{ix,iy}$.

The mask is still only permission to have nonzero entries. AD supplies
the actual numbers.

## 6. How P is used for compressed differentiation

Knowing an HVP does not tell us which columns can safely be combined. The
pattern supplies that missing information. The workflow is

```text
positions + model neighbor rule
           ↓
one-hop graph A → choose k → atom-pair pattern P^(k)
                                      ↓
                       Cartesian mask M_k and coloring
                                      ↓
                                seed matrix V
                                      ↓
                         HVPs of the full model: Y = HV
                                      ↓
                 use pattern + colors to reconstruct H
```

The seed matrix $V$ is called $S$ in the report's HVP example and in the
paper. Here we use $V$ for seeds and $M_k$ for the mask so the two objects
remain distinct:

| Object | Shape | What its entries mean |
| --- | --- | --- |
| $A$ | $N\times N$ | Direct neighbor connection |
| $P^{(k)}$ | $N\times N$ | Atom pair within the chosen hop limit |
| $M_k$ | $3N\times3N$ | Cartesian derivative position included in the pattern |
| $V$ | $3N\times n_c$ | Coordinate assigned to an HVP seed/color |
| $Y=HV$ | $3N\times n_c$ | Numerical sums of Hessian columns |
| $H$ | $3N\times3N$ | Actual second derivatives of energy |

Forward-over-reverse AD evaluates a Hessian-vector product (HVP):

$$H v = \left.\frac{d}{d\epsilon}\nabla_{\mathbf R}E(\mathbf R+\epsilon v)\right|_{\epsilon=0}.$$

This directional derivative is evaluated by AD, without taking a finite
displacement or constructing $H$ first. Dense reconstruction uses $3N$
basis-vector seeds. Coloring combines columns:
for color assignment $c(q)$, set $V_{qa}=1$ when $c(q)=a$ and zero otherwise.

$$Y=HV, \qquad Y_{pa}=\sum_{q:c(q)=a}H_{pq}.$$

For **ordinary column coloring**, columns may share a color only if their
permitted nonzero positions never overlap in a row. Under a sufficient
pattern, the sum then has only one possible contribution at each position
we read, so $H_{pq}=Y_{p,c(q)}$.

For example, temporarily give each atom only one coordinate and use
$P^{(1)}$ from the chain as a sufficient Hessian pattern. Columns 1 and 4
can share a color: their permitted entries occupy rows $\{1,2\}$ and
$\{3,4\}$, respectively. One HVP gives

$$H(e_1+e_4)=\begin{pmatrix}H_{11}\\H_{21}\\H_{34}\\H_{44}\end{pmatrix}.$$

The zeros guaranteed by the pattern let us assign each returned value to
its original column. Columns 2 and 3 each receive another color, so three
HVPs recover this four-column matrix. If we instead assume $P^{(2)}$,
columns 1 and 4 overlap in rows 2 and 3 and cannot share a color under
ordinary column coloring.

The paper uses **star coloring**, which also exploits $H_{pq}=H_{qp}$.
Some entries of $Y$ may contain ambiguous sums; recovery reads the
uncontaminated orientation of each symmetric pair. A star coloring is a
proper vertex coloring with no two-colored path on four distinct vertices.
Equivalently, each subgraph induced by two colors is a collection of stars,
where edges meet at a center. For an edge connecting a center and a leaf,
the leaf has just one neighbor of the other color, giving an unambiguous
compressed entry in that orientation.

Keep the graphs distinct. Hops are measured on the **input neighbor graph**
$A$. Star coloring acts on the **Hessian-pattern graph** built from
$P^{(k)}$, ignoring its diagonal as graph edges. For ordinary column
coloring, one instead builds a **column-conflict graph**, connecting columns
that overlap in a permitted row. Coloring the original neighbor graph alone
does not generally provide valid Hessian seeds.

The block structure allows atom colors $c_i$ to be lifted to Cartesian colors
$3c_i+d$, $d\in\{0,1,2\}$, using zero-based atom colors (Appendix D).
Thus $c_{\mathrm{atom}}$ atom colors become $n_c=3c_{\mathrm{atom}}$
Cartesian seed colors in this construction. An atom's $x,y,z$ coordinates
receive different colors; they are not all perturbed in the same seed.
With $n_c$ Cartesian colors, only $n_c$ HVP directions are needed.

## 7. What changes when we use too few hops?

At $k=K$, the pattern includes every possible Hessian coupling under the
chosen derivative convention. A conservative superset is also safe, although
it can require more colors. Additional nonlocal energy terms need their own
pattern; the local-message-passing bound alone does not cover them.

For $k<K$, truncated ASD discards distant couplings. Those couplings remain in
the full-energy HVPs and can also **contaminate retained entries** sharing a
color. Thus truncated reconstruction differs from masking a dense Hessian.

The ordinary-coloring chain example makes the problem explicit. Suppose we
group columns 1 and 4 using $P^{(1)}$, but the true model has a nonzero
two-hop entry $H_{24}$. The second component of that HVP is actually

$$[H(e_1+e_4)]_2=H_{21}+H_{24}.$$

Decompression based on $P^{(1)}$ assumes $H_{24}=0$ and assigns the whole
sum to $H_{21}$. It therefore both omits $H_{24}$ and changes a retained
entry. For example, if $H_{21}=2$ and $H_{24}=0.1$, the recovered value
is $2.1$. This example uses ordinary coloring for clarity; in star coloring,
omitted couplings can similarly contaminate entries expected to be clean
in the chosen recovery orientation.

For a three-layer PET model with $K=7$, choosing `hops=3` means building
$P^{(3)}$ and its coloring while still evaluating the full three-layer
model in each HVP. The expected benefit is fewer colors and fewer HVPs.
Its accuracy depends on the size of the neglected couplings, not just on
the fraction of matrix entries kept.

Let $M_k$ be the Cartesian mask and $\widetilde H_k$ the reconstructed matrix
before acoustic-sum-rule postprocessing, with entries outside $M_k$ set to
zero. The discarded and contaminated errors
have disjoint supports, with $\odot$ denoting entrywise multiplication:

$$\Delta H_{\mathrm{disc}}=-(\mathbf 1-M_k)\odot H,\qquad\Delta H_{\mathrm{cont}}=M_k\odot(\widetilde H_k-H).$$

$$\|\widetilde H_k-H\|_F^2=\|\Delta H_{\mathrm{disc}}\|_F^2+\|\Delta H_{\mathrm{cont}}\|_F^2.$$

Here $\mathbf 1$ is an all-ones matrix, not the identity, and
$\|B\|_F^2=\sum_{pq}|B_{pq}|^2$ adds the squared magnitudes of all entries.
The discarded error occupies positions outside the mask, and contamination
occupies positions inside it. Their positions do not overlap, so their squared
errors add without cross terms. The two errors vanish with a sufficient
pattern in exact arithmetic.

The paper finds rapid decay with hop distance. Most benchmark structures reach
0.1% heat-capacity accuracy at 300 K with two hops for MACE, three for PET-S,
and four for PET-XS; some zeolites need more (Section 5). These empirical results
provide no universal error bound or convergence guarantee for loaded MOF-5.

Compare increasing hop counts with an exact-pattern or feasible dense reference
at the same geometry and derivative convention. Check signed frequencies and
the final correction across temperatures: soft modes near the selection
threshold can change the mode count. Acoustic-sum-rule enforcement restores
translations but does not recover missing curvature. Periodic-cell convergence
is a separate requirement (Appendices A and E).

## 8. Where the computational saving comes from

Constructing $P^{(k)}$ uses connectivity, without evaluating second
derivatives. Coloring turns that connectivity information into a smaller
set of seeds. AD then evaluates the numerical derivatives for those seeds,
and decompression places the results in the matrix.

For an $O(N)$ energy evaluation at fixed density and architecture,
differentiated work is roughly $O(Nn_c)$ instead of $O(N^2)$ from $3N$
dense seeds. Pattern construction, coloring, compilation, batching, and
reconstruction add overhead; fewer HVPs do not guarantee an identical
factor of wall-time improvement. Smaller hop limits usually allow fewer
colors, but introduce the approximation described above.

In this project, the Hessian is ultimately stored as a dense matrix and
diagonalized for its full spectrum. That retains quadratic matrix storage
and cubic dense eigendecomposition cost. The sparse AD method accelerates
construction of the Hessian; its benefit must be assessed together with
the accuracy of the resulting frequencies and heat capacity.
