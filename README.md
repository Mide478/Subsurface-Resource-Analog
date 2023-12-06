# Subsurface-Resource-Analog

<!-- TABLE OF CONTENTS -->
<h2 id="table-of-contents"> :book: Table of Contents</h2>

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project"> ➤ About The Project</a></li>
    <li><a href="#prerequisites"> ➤ Prerequisites</a></li>
    <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
    <li><a href="#dataset"> ➤ Dataset</a></li>
    <li><a href="#roadmap"> ➤ Roadmap</a></li>
    <li><a href="#acknowledgements"> ➤ Acknowledgements</a></li>
    <li><a href="#contributors"> ➤ Contributors</a></li>
  </ol>
</details>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

<!-- ABOUT THE PROJECT -->
<h2 id="about-the-project"> :pencil: About The Project</h2>

<p align="justify"> 
 
* Spatial nonstationarity, ubiquitous in spatial settings, are generally not acocunted-for or even mitigated in deep learning (DL) applictaions for spatial phenomenon.

* We design a workflow to explore the impacts of nonstationarity on DL prediction performance:
  1. DL is trained with stationary SGS realizations of propoerty of interest (we use porosity for example) with variogram range labeled.
  2. Test the DL prediction performance using nonstationary realiztaions (investigate the impacts of nonstationarity).

* The benchmark results are obatined by training convolutional neural network (CNN) model, which is commonly used for computer vision (CV) tasks due to its performance in leanring spatial hierarchies of features. 

* Then we explore Vision Transformer (ViT) and Swin Transformer (SwinT) models for spatial nonstationarity mitigation. The original ViT and SwinT architectures are modifed for the predictive tasks (regression tasks).

* We found out self-attention networks can effectively help mitigate the spatial nonstatioanrity.
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

<!-- ROADMAP -->
<h2 id="roadmap"> :dart: Roadmap</h2>

<p align="justify"> 
  The goals of this project include the following:
<ol>
  <li>
    <p align="justify"> 
      Prepare the training data >>> statioanry SGS realizations (see Data Preparation.ipynb)
      Train the same models - Decision Tree, k Nearest Neighbors, and Random Forest using the preprocessed data obtained from topological data analysis and compare the
      performance against the results obtained by Weiss et. al.
    </p>
  </li>
  <li>
    <p align="justify"> 
      Train CNN model (see train_CNN.ipynb), train ViT, and SwinT model (see train_vision_transformers.ipynb) with training data. After the model is fully trained, test its prediction performance with testing data (nonstationary realizations).
    </p>
  </li>
<p align="justify">  
  
* CNN model is implemented with tensorflow packages. <b>train_CNN.ipynb</b> shows how to create your own CNN model and train it with your training data. 
      
* Vision transformers are implemented using Pytorch. <b>train_vision_transformers.ipynb</b> demonstrates the loading of ViT/SwinT architectures (<b>ViT.py</b> & <b>SwinT.py</b>), how to train the ViT/SwinT model. To visulzie the training progress, please couple it with tensorboard summary or wandb writer up to yourself.  
      
* Generally we need to train the DL models with a large number of training data. Here for easier demonstration, we randomly cretae single data (training data size =1) for both training and validation. In practical useage, you should train the model with data generated in Data Preparation.ipynb.
</p>
</ol>
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
