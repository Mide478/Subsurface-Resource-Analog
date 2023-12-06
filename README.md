# Subsurface-Resource-Analog

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#project-summary"> ➤ Project Summary</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#acknowledgements"> ➤ Acknowledgements</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PROJECT SUMMARY -->
<h2 id="project-sumary"> :pencil: Project Summary</h2>

<p align="justify"> 
 
* Determining subsurface resource analogs from mature subsurface resource datasets is essential for identifying and developing new prospects because critical information is often limited or absent initially. Conventionally, subsurface resource analogs are selected by geoscience and subsurface engineering domain experts. Furthermore, most subsurface datasets have high complexity, noise, and dimensionality, making interpretation and visualization difficult. Hence, adopting an objective machine learning workflow to support analog selection is beneficial. 

* We present a novel systematic and unbiased measure that accounts for spatial and multivariate data contributions using a novel dissimilarity metric and scoring metrics (group consistency score and pairwise similarity score), which measures within and between group similarities in lower dimensional spaces. The workflow consists of three steps, starting with inferential machine learning multidimensional scaling to obtain data representations in a lower dimensional space using the proposed novel dissimilarity metric. Next, inferential machine learning's density-based spatial clustering of applications with noise is applied in the lower dimensional space to identify analog clusters and spatial analogs. Afterward, the proposed scoring measures are used to quantify, summarize, and identify analogous data samples to assist with inferential diagnostics and resource exploration.

* We successfully identified and grouped analog clusters of well samples based on geological properties and cumulative gas production, showcasing the potential of our proposed workflow for practical use in the field.
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- PREREQUISITES -->
<h2 id="prerequisites"> :fork_and_knife: Prerequisites</h2>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) <br>
[![Made withJupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?style=for-the-badge&logo=Jupyter)](https://jupyter.org/try) <br>

<!--This project is written in Python programming language. <br>-->
The following open-source packages are mainly used in this project:
* Numpy
* Pandas
* Matplotlib
* Scikit-Learn

Please install other required packages detailed in the `requirements.txt` file and include custom-made `utils.py` containing functions in active working directory

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- :paw_prints:-->
<!-- FOLDER STRUCTURE -->
<h2 id="folder-structure"> :cactus: Folder Structure</h2>

    code
    .
    ├── Spatial_Weights_Determination.ipynb
    ├── Predictors_and_Response_Kriging_both_Spaces.ipynb
    ├── Comparative_Results_Analysis.ipynb  


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
<!-- DATASET -->
<h2 id="dataset"> :floppy_disk: Dataset</h2>
<p> 
  The dataset used for this demonstration is publicly available in <a href="[https://github.com/GeostatsGuy](https://github.com/GeostatsGuy/GeoDataSets/blob/master/)"> GeoDataSets: Synthetic Subsurface Data Repository as `12_sample_data.csv` </a> 
  
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ACKNOWLEDGEMENTS -->
<h2 id="acknowledgements"> :scroll: Acknowledgements</h2>
<p align="justify"> 
This work is supported by Equinor and Digital Reservoir Characterization Technology (DIRECT) Industry Affiliate Program at the University of Texas at Austin.
</p>


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- CONTRIBUTORS -->
<h2 id="contributors"> :scroll: Contributors</h2>

<p>  
  👩‍🎓: <b>Ademide O. Mabadeje</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>ademidemabadeje@utexas.edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/Mide478">@Mide478</a> <br>
  
  👨‍💻: <b>Jose J. Salazar</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>juliansalazarn@gmail.com</a> <br>
  
  👨‍💻: <b>Jesus Ochoa</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>jocho@equinor.com</a> <br>

  👩‍🏫: <b>Lean Garland</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>lega@equinor.com</a> <br>

  👨‍🏫: <b>Michael J. Pyrcz</b> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Email: <a>mpyrcz@austin.utexas.edu</a> <br>
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; GitHub: <a href="https://github.com/GeostatsGuy">@GeostatsGuy</a> <br>
</p>
<br>
