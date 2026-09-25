---
title: scatteringNet
emoji: 🧊
colorFrom: gray
colorTo: yellow
sdk: gradio
app_file: src/gradio/app.py
pinned: false
license: mit
short_description: "Occupancy fill: label inside points on a 3D mesh."
---

# scatteringNet — occupancy fill

Public Gradio demo. Upload an OBJ (or pick a sample) and **Run model** to label query points inside the solid.

[Code](https://github.com/PerryGu/scatteringNet)

[Video](https://youtu.be/vU45O0Mu0o4)

These sample meshes were not in the training catalog.

**Weights.** After this slim tree is on the Space, upload:

`models/2026-09-14_07-43-34_prim_extruded_nr45_knn24_n2048_n6/best.pt`

This folder is infer + Gradio only. It is not the training repo.
