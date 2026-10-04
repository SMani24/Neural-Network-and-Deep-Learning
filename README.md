# Neural Networks and Deep Learning

Coursework for Neural Networks and Deep Learning at the University of Tehran. The notebooks cover dense networks, computer vision, sequence modeling, transformers, generative models, and adversarial attacks. Assignment handouts, written reports, and reference papers are included alongside the code.

## Assignments

| Assignment | Notebooks | Documents |
|---|---|---|
| CA1 | [Credit-card fraud](assignments/ca1/q1/Q1.ipynb), [concrete-strength regression](assignments/ca1/q2/Q2.ipynb), [Adaline](assignments/ca1/q3/Q3.ipynb), [MNIST autoencoders](assignments/ca1/q4/Q4.ipynb) | [Handout](assignments/ca1/specifications/assignment.pdf) · [Report](assignments/ca1/reports/report.pdf) |
| CA3 | [CamVid segmentation with Fast-SCNN](assignments/ca3/q1/Q1.ipynb) | [Handout](assignments/ca3/specifications/assignment.pdf) |
| CA4 | [Clinical time-series analysis and recurrent models](assignments/ca4/q2/Q2.ipynb) | [Handout](assignments/ca4/specifications/assignment.pdf) |
| CA5 | [Tomato disease classification: CNNs and ViT](assignments/ca5/q1/Q1.ipynb), [CLIP, LoRA, and adversarial adaptation](assignments/ca5/q2/Q2.ipynb) | [Handout](assignments/ca5/specifications/assignment.pdf) · [Report](assignments/ca5/reports/report.pdf) |
| CA6 | [GAN domain adaptation](assignments/ca6/q1/Q1.ipynb), [EndoVAE reconstruction](assignments/ca6/q2/Q2.ipynb) | [Handout](assignments/ca6/specifications/assignment_fa.pdf) · [English handout](assignments/ca6/specifications/assignment_en.pdf) · [Report](assignments/ca6/reports/report.pdf) |
| Additional assignment (CAe) | [ResNet/ViT experiments and adversarial training](assignments/cae/q1/Q1_extended.ipynb), [Persian captioning data preparation](assignments/cae/q2/Q2.ipynb) | [Handout](assignments/cae/specifications/assignment_fa.pdf) · [English handout](assignments/cae/specifications/assignment_en.pdf) · [Report](assignments/cae/reports/report.pdf) |

CA2 is not present. Some tasks appear only in the handouts, and some notebooks include unfinished experiments.

## Layout

- `assignments/`: question folders (`q1`, `q2`, etc.), with handouts in `specifications/`, written work in `reports/`, and papers in `references/`.
- [lectures/](lectures/): slides for Chapters 2–7.
- [archives/](archives/): packaged assignments and dataset archives, plus copies previously stored at assignment roots. Archive contents have been left intact.

Alternate notebook versions remain in their question folders. Local datasets, checkpoints, and outputs stay beside the notebooks that use them; most are excluded from Git and may need to be obtained separately after cloning.

## Working with the notebooks

Use the question folder as your working directory so relative paths resolve. For example:

```bash
cd assignments/ca6/q1
jupyter lab
```

The main dependencies are Python, Jupyter, TensorFlow/Keras, and PyTorch/torchvision. Individual notebooks also use libraries such as scikit-learn, Hugging Face Datasets/Transformers, PEFT, statsmodels, and torchmetrics. Check their imports and dataset paths before running them; there is no shared environment file.

Saved outputs are retained, including failed or interrupted runs. Rerunning a notebook can require substantial training time and model downloads. The small Python exports do not always contain the full notebook workflow.
