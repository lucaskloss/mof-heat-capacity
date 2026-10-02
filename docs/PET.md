# Point Edge Transformers

Sergey N. Pozdnyakov and Michele Ceriotti, *Smooth, exact rotational
symmetrization for deep learning on point clouds*, NeurIPS 2023.
The supplied [PET.pdf](PET.pdf) is arXiv version 3 (6 February 2024);
page references below use that version.

The paper introduces **PET**, a Cartesian transformer for atomistic prediction,
and **ECSE**, an ensemble of local coordinate frames providing exact rotational
equivariance. PET-MAD is a later fitted potential. Implementation and checkpoint
notes are in [HESSIANS.md](../scripts/properties/HESSIANS.md); sparse-AD theory
is in [SADMOF.md](SADMOF.md). Equations below include the paper's construction
and supporting mathematical derivations.

## Energy and symmetry

An extensive potential sums local contributions and species baselines:

$$U_\theta(\mathbf R,\mathbf h,\mathbf Z)=\sum_i b_{Z_i}+\sum_i\varepsilon_i(\mathcal A_i;\theta).$$

For a proper rotation $Q$, translation $\mathbf a$, and atom permutation
$\pi$, the desired energy symmetries are

$$U_\theta(Q\mathbf R+\mathbf a,Q\mathbf h,\mathbf Z)=U_\theta(\mathbf R,\mathbf h,\mathbf Z),\qquad U_\theta(\pi\mathbf R,\mathbf h,\pi\mathbf Z)=U_\theta(\mathbf R,\mathbf h,\mathbf Z).$$

Rotate the periodic cell with the atoms. Exact energy invariance implies force
covariance and the Hessian transformation

$$\mathbf F_i(Q\mathbf R,Q\mathbf h)=Q\mathbf F_i(\mathbf R,\mathbf h),\qquad H'_{i\alpha,j\beta}=\sum_{\gamma\delta}Q_{\alpha\gamma}H_{i\gamma,j\delta}Q_{\beta\delta}.$$

Forces derived from a twice-differentiable scalar energy are conservative:

$$\frac{\partial F_{i\alpha}}{\partial R_{j\beta}}=\frac{\partial F_{j\beta}}{\partial R_{i\alpha}}=-H_{i\alpha,j\beta}.$$

Symmetry and physical accuracy require separate validation. PET uses full
Cartesian geometry with unconstrained attention; rotational augmentation
encourages approximate invariance. ECSE enforces symmetry through local frames
(Sections 1–5).

## Local environments and messages

A periodic neighbor includes its image offset $\mathbf n\in\mathbb Z^3$:

$$\mathbf r_{ij\mathbf n}=\mathbf R_j+\mathbf h\mathbf n-\mathbf R_i,\qquad \lVert\mathbf r_{ij\mathbf n}\rVert<R_c.$$

PET retains distinct directed-edge messages and processes incoming messages
jointly. Smooth cutoffs remove departing neighbors. A sufficient condition
for continuous second derivatives is

$$f_c(r)=1\ \text{in the inner region},\qquad f_c(R_c)=f_c'(R_c)=f_c''(R_c)=0.$$

An illustrative quintic taper between $R_s$ and $R_c$ is

$$s=\frac{r-R_s}{R_c-R_s},\qquad f_c(r)=\begin{cases}1,&r\le R_s,\\1-10s^3+15s^4-6s^5,&R_s<r<R_c,\\0,&r\ge R_c.\end{cases}$$

This taper is an example rather than a universal PET cutoff. In the original
architecture, a neighbor token combines geometry, species, and an incoming
message (Appendix A, Eq. 5):

$$\mathbf p_{ij}^{(\ell)}=\operatorname{SiLU}\!\left(\mathbf W_r^{(\ell)}\mathbf r_{ij}+\mathbf b_r^{(\ell)}\right),\qquad \mathbf t_{ij}^{(\ell)}=M_\ell\!\left[\mathbf m_{j\to i}^{(\ell-1)}\Vert E_n^{(\ell)}(Z_j)\Vert\mathbf p_{ij}^{(\ell)}\right].$$

Each ingredient has width $d_{\mathrm{PET}}$. The MLP compresses the
concatenation; the first block omits incoming messages. A separate central
token is $\mathbf t_{i0}^{(\ell)}=E_c^{(\ell)}(Z_i)$.

## Local attention and energy readout

For token matrix $X$, each attention head projects queries, keys, and values:

$$Q_h=XW_h^Q,\qquad K_h=XW_h^K,\qquad V_h=XW_h^V,$$

With $c_0=1$ for the central token and $c_j=f_c(r_{ij})$ for neighbors,

$$\alpha_{ab}=\frac{\exp(\mathbf q_a^\mathsf T\mathbf k_b/\sqrt{d_k})\,c_b}{\sum_c\exp(\mathbf q_a^\mathsf T\mathbf k_c/\sqrt{d_k})\,c_c},\qquad \mathbf y_a=\sum_b\alpha_{ab}\mathbf v_b.$$

Cutoff weights enter before normalization. Heads are combined as

$$\operatorname{MHA}(X)=\operatorname{Concat}(Y_1,\ldots,Y_H)W^O,$$

A schematic pre-normalization layer is

$$X'=X+\operatorname{MHA}(\operatorname{LN}X),\qquad X''=X'+\operatorname{FFN}(\operatorname{LN}X').$$

Shared token operations preserve permutation equivariance when masks and
cutoff weights follow the same permutation $P$:

$$\operatorname{Attention}(PX)=P\operatorname{Attention}(X).$$

A symmetric readout gives permutation-invariant energy. Joint attention on
Cartesian vectors can represent angles and higher-body geometry:

$$\mathbf r_{ij}^{\mathsf T}\mathbf r_{ik}=r_{ij}r_{ik}\cos\theta_{jik}.$$

Transformer depth mixes one environment; message-passing depth $L$ extends
its graph receptive field. For bounded neighborhoods, total cost is
approximately linear in atom count (Section 6).

The summed-edge readout collects central and cutoff-weighted edge predictions
across blocks (Appendix A, Eqs. 6–7):

$$U_\theta=\sum_i b_{Z_i}+\sum_{\ell=1}^{L}\sum_i\left[H_c^{(\ell)}(\mathbf x_{i0}^{(\ell)})+\sum_{j\in\mathcal A_i}f_c(r_{ij})H_n^{(\ell)}(\mathbf x_{ij}^{(\ell)})\right].$$

Edge predictions depend on their environments. For readout terms $u_a$, forces
and Hessians couple every atom influencing those terms:

$$\mathbf F_k=-\sum_a\frac{\partial u_a}{\partial\mathbf R_k},\qquad H_{k\alpha,l\beta}=\sum_a\frac{\partial^2u_a}{\partial R_{k\alpha}\partial R_{l\beta}}.$$

## ECSE: exact rotational symmetrization

For two noncollinear neighbor directions, construct a right-handed frame:

$$\mathbf e_1=\widehat{\mathbf r}_{ij},\qquad \mathbf e_2=\frac{\widehat{\mathbf r}_{ik}-(\widehat{\mathbf r}_{ik}\!\cdot\!\mathbf e_1)\mathbf e_1}{\left\lVert\widehat{\mathbf r}_{ik}-(\widehat{\mathbf r}_{ik}\!\cdot\!\mathbf e_1)\mathbf e_1\right\rVert},\qquad \mathbf e_3=\mathbf e_1\times\mathbf e_2,\qquad Q_{ijk}=[\mathbf e_1\ \mathbf e_2\ \mathbf e_3].$$

Evaluate the backbone in each frame and average scalar predictions:

$$\varepsilon_i^{\mathrm{ECSE}}=\frac{\sum_{j\ne k}w_{ijk}\,\varepsilon_i^0(Q_{ijk}^{\mathsf T}\mathcal A_i)}{\sum_{j\ne k}w_{ijk}}.$$

Under a global rotation $S$, $Q'_{ijk}=SQ_{ijk}$, hence

$${Q'}_{ijk}^{\mathsf T}(S\mathbf r)=Q_{ijk}^{\mathsf T}S^{\mathsf T}S\mathbf r=Q_{ijk}^{\mathsf T}\mathbf r.$$

Local-frame coordinates are invariant. Vector and tensor predictions are
transformed back before averaging:

$$\mathbf v_{ijk}=Q_{ijk}\mathbf v^0_{ijk},\qquad T_{ijk}=Q_{ijk}T^0_{ijk}Q_{ijk}^{\mathsf T},$$

They obey $\mathbf v'=S\mathbf v$ and $T'=STS^{\mathsf T}$.
This construction enforces proper-rotation symmetry; reflection symmetry
requires additional treatment.

Smooth frame weights suppress cutoff neighbors and nearly collinear pairs:

$$w_{ijk}=f_c(r_{ij})f_c(r_{ik})q_c\!\left(\lVert\widehat{\mathbf r}_{ij}\times\widehat{\mathbf r}_{ik}\rVert^2\right).$$

Fully collinear environments need special handling. With frame index $a$,
$p_a=w_a/\sum_b w_b$, and $\bar\varepsilon=\sum_a p_a\varepsilon_a$,
differentiation for positive active weights gives

$$\frac{\partial\bar\varepsilon}{\partial q}=\sum_a p_a\frac{\partial\varepsilon_a}{\partial q}+\sum_a p_a(\varepsilon_a-\bar\varepsilon)\frac{\partial\log w_a}{\partial q}.$$

Forces must include derivatives of both the frames and normalized weights.
Tight selection can amplify these derivatives, affecting Hessians (Section 5;
Appendices F.1–F.7). ECSE is separate from PET's later adaptive neighbor cutoff
and is not used by this project's fitted models.

## Training and evidence

Species baselines are fitted over reference structures $s$, where $n_{sZ}$
counts species $Z$:

$$\mathbf b^*=\underset{\mathbf b}{\operatorname{argmin}}\sum_s\left(E_s^{\mathrm{ref}}-\sum_Z n_{sZ}b_Z\right)^2,$$

The network learns residual energies. Coordinate-independent baselines leave
forces and Hessians unchanged. A schematic energy–force objective uses moving
validation-error scales:

$$\mathcal L=\lambda_E\frac{\operatorname{MSE}(U_\theta,U^{\mathrm{ref}})}{\overline{\operatorname{MSE}}_{E,\mathrm{val}}}+\lambda_F\frac{\operatorname{MSE}(-\nabla_{\mathbf R}U_\theta,\mathbf F^{\mathrm{ref}})}{\overline{\operatorname{MSE}}_{F,\mathrm{val}}},$$

Rotational augmentation transforms samples to $(Q\mathbf R,Q\mathbf F)$.
The paper demonstrates expressive learning on water, molecular collisions,
alloys, and other benchmarks (Section 7; Appendix C). Those results do not
establish methane–framework accuracy or heat-capacity uncertainty for MOF-5.

## Harmonic quantum correction

At a fixed-cell local minimum $\mathbf R_0$,

$$U(\mathbf R_0+\Delta\mathbf R)\approx U(\mathbf R_0)+\frac12\Delta\mathbf R^{\mathsf T}H\Delta\mathbf R,\qquad H=\left.\nabla_{\mathbf R}^{2}U\right|_{\mathbf R_0},$$

Forward-over-reverse AD evaluates curvature along a direction without forming
the full Hessian:

$$H\mathbf v=\left.\frac{\mathrm d}{\mathrm d\epsilon}\nabla_{\mathbf R}U(\mathbf R+\epsilon\mathbf v)\right|_{\epsilon=0}.$$

After reconstruction, mass weighting gives the normal modes:

$$D_{i\alpha,j\beta}=\frac{H_{i\alpha,j\beta}}{\sqrt{m_i m_j}},\qquad D\mathbf e_k=\omega_k^2\mathbf e_k.$$

For retained positive modes, $x_k=\hbar\omega_k/(k_BT)$, and

$$C_{V,k}^{\mathrm{qn}}=k_B\frac{x_k^2e^{x_k}}{(e^{x_k}-1)^2},\qquad C_{V,k}^{\mathrm{cl}}=k_B,$$

Each quantum contribution approaches $k_B$ as $x_k\to0$ and zero as
$x_k\to\infty$. For one validated mode set $\mathcal M$,

$$\Delta C_V^{\mathrm{har}}(T)=\sum_{k\in\mathcal M}\left[C_{V,k}^{\mathrm{qn}}(T)-k_B\right],$$

The loaded-system hybrid approximation is

$$C_P^{\mathrm{approx}}(T)=\frac{\mathrm d}{\mathrm dT}\left\langle U+K+P_{\mathrm{ext}}V\right\rangle_{NPT}+\Delta C_V^{\mathrm{har}}(T).$$

Classical NPT samples anharmonic motion. Adding a fixed-cell $C_V$ correction
to its $C_P$ derivative is an approximation. Validation must cover matching
energy/force conventions, rotations of atoms and cell, Hessian and cell-size
convergence, stable minima, and the relevant host–guest chemistry.
