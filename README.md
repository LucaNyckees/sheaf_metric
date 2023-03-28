# Integral Sheaf Metrics

This project is supervised by Nicolas Berkouk and Kathryn Hess Bellwald, from the [Laboratory of Topology and Neuroscience at EPFL](https://www.epfl.ch/labs/hessbellwald-lab/).

## Table of Content

- [People](#people)
- [Description](#description)
- [Virtual Environment](#virtual-environment)
- [Project Organization](#project-organization)
- [Streamlit Web App](#streamlit)
- [Related Articles and Useful References](#refs)
- [Interesting Material 🔍](#material)

## People

Nicolas Berkouk : [EPFL profile](https://people.epfl.ch/nicolas.berkouk), [Personal site](https://nberkouk.github.io/)<br />
Luca Nyckees : [EPFL profile](https://people.epfl.ch/luca.nyckees)

## Mathematical Context

Topological data analysis emerged, as a field of mathematics, as a powerful tool to study the shape of data. It has found many applications in domains like computational neuroscience, machine learning and statistical data analysis. The most important concept in this field is the notion of persistent homology, allowing one to associate (stable and complete) topological computational invariants to point cloud data and other spaces. The mathematical setting of persistent homology is very well understood. There is a natural generalization to higher dimensions, called multi-parameter persistent homology, that provides a much greater challenge, as the related algebraic objects to deal with are harder to manipulate.

## Description

The class of integral sheaf metrics is an esemble of distances between multi-parameter persistence modules that successfully tackles the issues encountered with the matching distance. Through computational geometry and optimization machine learning, we provide a way to efficiently compute the so-called linear integral sheaf metric. We also present applications to data classification - the input to standard clustering algorithms is a distance matrix encoding pairwise linear integral sheaf metrics between multi-parameter persistence modules.

<p align="center">
<img width="800" alt="figure" src="https://github.com/LucaNyckees/topology-metric/blob/main/def_ISM.png">
</p>

## Virtual Environment

### Virtual environment

```
python3.11 -m venv venv
source venv/bin/activate
```

### Install packages

```
pip install -r requirements.txt
```

### Exit virtual environment

```
deactivate
```

## Project Organization

---

    ├── README.md          -- Top-level README.
    │
    ├── notebooks          -- Jupyter notebooks.
    │
    ├── articles           -- Related articles and useful references.
    │
    ├── reports            -- Notes and report (Latex, pdf).
    │ 
    ├── figures            -- Optional graphics and figures to be included in the report.
    │
    ├── requirements.txt   -- Requirements file for reproducibility.
    └── src                -- Project source code.

---
