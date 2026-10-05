# Q1 notebook versions

Start with [Q1_extended.ipynb](Q1_extended.ipynb) for the most complete write-up. It includes the final comparison table and theoretical answers, but also retains a failed run. [Q1_recovered_split.ipynb](Q1_recovered_split.ipynb) adds a separate scratch-ViT defense run that reaches ten training epochs; its saved output stops during evaluation.

The recovered files came from `New Found Files`. Their contents do not establish a reliable save order, so these names describe the workflow rather than claiming a final version.

| Notebook | What it preserves |
|---|---|
| [Q1.ipynb](Q1.ipynb) | Existing earlier workflow, including interrupted training and a CUDA error. |
| [Q1_extended.ipynb](Q1_extended.ipynb) | Most complete existing write-up and manually consolidated results. |
| [Q1_recovered_split.ipynb](Q1_recovered_split.ipynb) | Former `Q1(1)(1).ipynb`; separate ResNet, pretrained-ViT, and scratch-ViT defense runs. |
| [Q1_recovered_split_variant.ipynb](Q1_recovered_split_variant.ipynb) | Former `Q1(1).ipynb`; differently initialized results list and unfinished scratch defense output. |
| [Q1_recovered_split_quiet.ipynb](Q1_recovered_split_quiet.ipynb) | Former `Q1(4).ipynb`; removes one diagnostic print from the preceding variant. |
| [Q1_recovered_combined.ipynb](Q1_recovered_combined.ipynb) | Former `Q1.ipynb`; separates ResNet defense from a combined ViT loop. |
| [Q1_recovered_combined_plot_error.ipynb](Q1_recovered_combined_plot_error.ipynb) | Former `Q1(3).ipynb`; same code as the combined notebook, with different outputs and a plotting error. |
| [Q1_recovered_single_loop.ipynb](Q1_recovered_single_loop.ipynb) | Former `Q1(2).ipynb`; all four defense models in one loop, ending in a CUDA error. |

All notebooks are kept unchanged. Run them from this folder so `./data` resolves to the existing datasets. Saved outputs can reflect earlier code or an interrupted session.
