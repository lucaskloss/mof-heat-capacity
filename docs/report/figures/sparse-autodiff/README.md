# Illustrations from the ICLR sparse autodiff article

Source: [An Illustrated Guide to Automatic Sparse Differentiation](https://iclr-blogposts.github.io/2025/blog/sparse-autodiff/), Adrian Hill, Guillaume Dalle, and Alexis Montoison (2025).

Downloaded on 2026-10-12. Original SVGs are preserved; matching PDFs were converted with `rsvg-convert` for LaTeX inclusion. The report credits the authors and cites the article in every caption. These are illustrative Jacobians, not measured MOF Hessians.

| Files (SVG and PDF) | Original figure | Source asset |
| --- | --- | --- |
| `sparsity_pattern` | 12a | [SVG](https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparsity_pattern.svg) |
| `coloring` | 12b | [SVG](https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/coloring.svg) |
| `sparse_ad_forward_full` | 10a | [SVG](https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_full.svg) |
| `sparse_ad_forward_decompression` | 10b | [SVG](https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/sparse_ad_forward_decompression.svg) |
| `colored_graph` | 17 | [SVG](https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/colored_graph.svg) |
| `colored_graph_infeasible` | 19 | [SVG](https://iclr-blogposts.github.io/2025/assets/img/2025-04-28-sparse-autodiff/colored_graph_infeasible.svg) |

Conversion command, applied to each SVG:

```bash
rsvg-convert -f pdf -o name.pdf name.svg
```
