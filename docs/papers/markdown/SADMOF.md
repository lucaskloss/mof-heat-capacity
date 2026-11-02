# SADMOF: sparse Hessians and harmonic heat capacity

Marcel F. Langer, Adrian Hill, and Michele Ceriotti, *Truncated automatic sparse
differentiation for machine learning interatomic potentials*,
[supplied preprint](SADMOF.pdf). Section 4 describes the method; Appendices C,
D, and H derive the reach, block coloring, and truncation error.
Commands, library calls, parameters, archives, and debugging are documented
beside the scripts in [HESSIANS.md](../scripts/properties/HESSIANS.md).

## Sparsity from locality

For $N$ atoms, the Cartesian force-constant matrix is

$$H_{i\alpha,j\beta} = \frac{\partial^2 E}{\partial R_{i\alpha}\,\partial R_{j\beta}}. $$

Two atoms can couple when they influence the same energy term. With $L$
message-passing layers, a node readout has maximum Hessian reach $K=2L$;
PET's edge readout spans neighborhoods of adjacent atoms, giving $K=2L+1$.
These bounds assume the stated architecture and derivative convention.

| Model in the paper | $L$ | Exact reach $K$ |
| --- | --- | --- |
| MACE-MP-0 medium | 2 | 4 |
| PET-MAD-1.5 XS | 2 | 5 |
| PET-MAD-1.5 S | 3 | 7 |

For undirected input-graph adjacency $A$, Boolean arithmetic gives the
atom-pair pattern through $k$ hops:

$$P^{(k)} = I \lor A \lor A^2 \lor \cdots \lor A^k.$$

The identity includes on-site blocks; each allowed pair expands to a
$3\times3$ Cartesian block. At $k=K$, entries outside the pattern are
structurally zero. A conservative superset is safe but may require more colors.
Additional nonlocal energy terms need their own pattern. For PET, the paper
stops adaptive-cutoff derivatives and uses the selected neighbor graph.

## Compressed differentiation

Forward-over-reverse AD evaluates a Hessian-vector product (HVP):

$$H v = \left.\frac{d}{d\epsilon}\nabla_{\mathbf R}E(\mathbf R+\epsilon v)\right|_{\epsilon=0}.$$

Dense reconstruction uses $3N$ basis-vector seeds. Coloring combines columns:
for color assignment $c(q)$, set $V_{qa}=1$ when $c(q)=a$ and zero otherwise.

$$Y=HV, \qquad Y_{pa}=\sum_{q:c(q)=a}H_{pq}.$$

Ordinary column coloring forbids same-color nonzeros in a row, so
$H_{pq}=Y_{p,c(q)}$. **Star coloring** uses Hessian symmetry to recover each
entry from either orientation. It forbids two-colored paths on four vertices;
each two-color subgraph is a collection of stars. At least one orientation of
each off-diagonal entry then has an uncontaminated compressed value.

The block structure allows atom colors $c_i$ to be lifted to Cartesian colors
$3c_i+d$, $d\in\{0,1,2\}$ (Appendix D). With $n_c$ colors, only $n_c$
HVP directions are needed. For an $O(N)$ energy evaluation at fixed density,
differentiated work is $O(Nn_c)$ instead of $O(N^2)$; setup and reconstruction
add overhead. Full-spectrum analysis still requires matrix storage and
an eigendecomposition.

## Truncation error and convergence

For $k<K$, truncated ASD discards distant couplings. Those couplings remain in
the full-energy HVPs and can also **contaminate retained entries** sharing a
color. Thus truncated reconstruction differs from masking a dense Hessian.

Let $M_k$ be the Cartesian mask and $\widetilde H_k$ the reconstructed matrix
before acoustic-sum-rule postprocessing. The discarded and contaminated errors
have disjoint supports, with $\odot$ denoting entrywise multiplication:

$$\Delta H_{\mathrm{disc}}=-(1-M_k)\odot H, \qquad \Delta H_{\mathrm{cont}}=M_k\odot(\widetilde H_k-H).$$

$$\|\widetilde H_k-H\|_F^2=\|\Delta H_{\mathrm{disc}}\|_F^2+\|\Delta H_{\mathrm{cont}}\|_F^2.$$

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

## Frequencies and heat capacity

At a fixed-cell minimum, mass weighting gives the dynamical matrix:

$$D_{i\alpha,j\beta} = \frac{H_{i\alpha,j\beta}}{\sqrt{m_i m_j}}, $$

Its eigenvalues are $\omega^2$. Negative values indicate imaginary modes and
must remain visible when validating a minimum. Uniform translations should
supply three zero modes.

For each retained positive-frequency mode,

$$C_{V,k}^{\mathrm{qn}}(T) = k_B\frac{x_k^2 e^{-x_k}}{(1-e^{-x_k})^2}, \qquad x_k=\frac{h c\tilde\nu_k}{k_B T}. $$

Sum over modes and divide by cell mass for gravimetric $C_V$ in
$J g^{-1} K^{-1}$. Each mode approaches $k_B$ at high temperature and freezes
out at low temperature. Use the same validated mode set in both terms of

$$\Delta C^{\mathrm{har}}(T) = C_{V,\mathrm{qn}}^{\mathrm{har}}(T) - C_{V,\mathrm{cl}}^{\mathrm{har}}. $$

The project's hybrid approximation adds the loaded-system correction to the
classical NPT enthalpy derivative:

$$C_P^{\mathrm{approx}}(T) = \frac{d\langle E_{\mathrm{tot}}+P_{\mathrm{ext}}V\rangle}{dT} + \Delta C^{\mathrm{har}}(T). $$

Classical MD supplies anharmonic motion; the fixed-cell harmonic correction
replaces the classical statistics of those modes with quantum statistics.
The empty-framework Hessian is a separate reference. Using a $C_V$ correction
with an NPT $C_P$ derivative remains an approximation requiring validation.
