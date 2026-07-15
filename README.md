## Quantum Time Series Model Expressivity and Trainability in PennyLane+PyTorch

### Our team
- Marcin
- Marysia
- Miron
- Adrian
- Sebastian
- Jacob
- Bartek

### Aims
This project establishes a formal assessment framework to systematically evaluate two pivotal attributes of quantum model design: 
expressivity (the capacity of a model to represent complex data distributions) and 
trainability (the susceptibility of the model to optimization challenges, such as barren plateaus). 
While increasing circuit depth, width, and entanglement capacity typically enhances expressivity, 
it often exponentially degrades the trainability of the cost landscape.

To map this complex trade-off, this research introduces a quantitative pipeline analyzing the impact of key structural hyperparameters: 
circuit width, layer depth, parameter density, and entanglement topology. 

Expressivity will be formally profiled using the following methods:
- Krylov metrics of expressivity (Miron);
- Haar Integration and KL divergence expressivity (Marcin).
  
Trainability will be rigorously assessed via:
- capacity-based measures like the Effective Dimension (Adrian);
- the spectral analysis of the Fisher Information Matrix (FIM) (Adrian);
- gradient variance scaling bounds.

A formal mathematical framework will also be proposed:
- Mathematical foundations of expressivity and trainability (Marysia)

By decoupling and independently modulating these architectural variables across standard ansatz families, 
this project aims to transition quantum circuit design from heuristic trial-and-error to a predictable, mathematically grounded engineering discipline. 
The final deliverable will provide a predictive blueprint for identifying optimal "sweet spots" in ansatz design—maximizing expressive power while preserving robust optimization trajectories.

### Folders
*We'll need some common utilities, I suggest to keep them as .py files in a directory.*
- examples: team members examples and supporting notebooks
- notebooks: notebooks delivering various project aspects
- install: advice on installing the PennyLane environment
- legacy: recent versions of archived files
- logs: this folder may be created to hold saved data, training history, plots, etc.
- src: various Python libraries to support the project
- utils: a collection of Python general purpose utilities, e.g.
  - Charts.py - functions plotting time-series data (fancy and flexible)
  - Files.py - functions saving time-series and support data to disk
  - Tools.py - some odd collection of utilities, including extras for PennyLane
  - Window.py - functions creating and managing sliding windows (making, splitting, etc.)

### Requirements
- Set up a virtual environment with **venv** or **anaconda** for Python 3.11 and activate it
- Then install all software using **requirements.txt** file (available here):
    - pip install -r \<place-you-saved-it\>/requirements.txt
- Or install by hand by following these instructions:
    - pip install pennylane==0.40.0 pennylane-lightning==0.40.0 (PennyLane for CPU)
    - pip install scikit-learn==1.6.1 pandas==2.2.3 (ML)
    - pip install matplotlib==3.10.1 plotly==6.0.0 seaborn==0.13.2 pillow==11.1.0 (plots and images)
    - pip install jupyter==1.1.1 jupyterlab==4.3.5 (running jupyter notebooks)
    - pip install kagglehub==0.3.10 ucimlrepo==0.0.7 (data access)
    - pip install pdflatex (optionally to plot and export some plots and tables to latex)
    - install [PyTorch](https://pytorch.org/get-started/locally/), as per web site instructions, also add:<br>
      pip install torchsummary torcheval torchmetrics

The **requirements.txt** file was tested for installation on 
Ubuntu 22.04-24.04, Windows 11 and MacOS Sequoia 15.3.1 (with M3 procesor).

### Project Discord
All changes in this repository will be announced in the project Disord channel **expressivity-github-notifications**.

### License
This project is licensed under the [GNU General Public License v3](./LICENSE).
The GPL v3 license requires attribution for modifications and derivatives, ensuring that users know which versions are changed and to protect the reputations of original authors.