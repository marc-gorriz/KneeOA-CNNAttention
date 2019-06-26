# Assessing Knee OA Severity with CNN attention-based end-to-end architectures

| ![Marc Górriz][MarcGorriz-photo]  | ![Joseph Antony][JosephAntony-photo] | ![Kevin McGuinness][KevinMcGuinness-photo]  | ![Xavier Giro-i-Nieto][XavierGiro-photo]  | ![Noel E. O'Connor][NoelOConnor-photo]  |
|:-:|:-:|:-:|:-:|:-:|
| [Marc Górriz][MarcGorriz-web]  | [Joseph Antony][JosephAntony-web] | [Kevin McGuinness][KevinMcGuinness-web] | [Xavier Giro-i-Nieto][XavierGiro-web] | [Noel E. O'Connor][NoelOConnor-web] |

[MarcGorriz-web]: https://www.linkedin.com/in/marc-górriz-blanch-74501a123/
[XavierGiro-web]: https://imatge.upc.edu/web/people/xavier-giro
[JosephAntony-web]: https://imatge.upc.edu/web/people/xavier-giro
[KevinMcGuinness-web]: https://www.insight-centre.org/users/kevin-mcguinness
[NoelOConnor-web]: https://www.insight-centre.org/users/noel-oconnor

[MarcGorriz-photo]: https://github.com/marc-gorriz/KneeOA-CNNAttention/blob/master/authors/MarcGorriz.jpg
[XavierGiro-photo]: https://github.com/marc-gorriz/KneeOA-CNNAttention/blob/master/authors/XavierGiro.jpg
[JosephAntony-photo]: https://github.com/marc-gorriz/KneeOA-CNNAttention/blob/master/authors/JosephAntony.jpg
[KevinMcGuinness-photo]: https://github.com/marc-gorriz/KneeOA-CNNAttention/blob/master/authors/KevinMcGuinness.jpg
[NoelOConnor-photo]: https://github.com/marc-gorriz/KneeOA-CNNAttention/blob/master/authors/NoelOConnor.jpg

A joint collaboration between:

| ![logo-insight] | ![logo-dcu] | ![logo-gpi] |
|:-:|:-:|:-:|
| [IRIT Vortex Group][insight-web] | [INP Toulouse - ENSEEIHT][dcu-web] | [UPC Image Processing Group][gpi-web] |

[insight-web]: https://www.insight-centre.org/ 
[dcu-web]: http://www.dcu.ie/
[upc-web]: http://www.upc.edu/?set_language=en/
[etsetb-web]: https://www.etsetb.upc.edu/en/ 
[gpi-web]: https://imatge.upc.edu/web/ 


[logo-insight]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/insight.jpg "Insight Centre for Data Analytics"
[logo-dcu]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/dcu.png "Dublin City University"
[logo-upc]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/upc.jpg "Universitat Politecnica de Catalunya"
[logo-etsetb]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/etsetb.png "ETSETB TelecomBCN"
[logo-gpi]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/gpi.png "UPC Image Processing Group"

## Abstract

This work proposes a novel end-to-end convolutional neural network (CNN) architecture to automatically quantify the severity of knee osteoarthritis (OA) using X-Ray images, which incorporates trainable attention modules acting as unsupervised fine-grained detectors of the region of interest (ROI). The proposed attention modules can be applied at different levels and scales across any CNN pipeline helping the network to learn relevant attention patterns over the most informative parts of the image at different resolutions. We test the proposed attention mechanism on existing state-of-the-art CNN architectures as our base models, achieving promising results on the benchmark knee OA datasets from the osteoarthritis initiative (OAI) and multicenter osteoarthritis study (MOST).

![system-fig]

[system-fig]: https://raw.githubusercontent.com/marc-gorriz/KneeOA-CNNAttention/master/figs/system_diagram.png

---

## How to use

### Dependencies

The model is implemented in [Keras](https://github.com/fchollet/keras/tree/master/keras), which at its time is developed over [TensorFlow](https://www.tensorflow.org). Also, this code should be compatible with Python 3.4.2.

```
pip install -r https://github.com/marc-gorriz/KneeOA-CNNAttention/blob/master/requeriments.txt
```

### Launch an experiment
* Make a new configuration file based on the available templates and save it into the ```config``` directory.
Make sure to launch all the processes over GPU. On this project there was used an NVIDIA GTX Titan X.

* To train a new model, run  ```python train.py --config_path config/[config file].py```.

## Acknowledgements

We would like to especially thank Albert Gil Moreno from our technical support team at the Image Processing Group at the UPC.

| ![AlbertGil-photo]  |
|:-:|
| [Albert Gil](AlbertGil-web)   |

[AlbertGil-photo]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/authors/AlbertGil.jpg "Albert Gil"

[AlbertGil-web]: https://imatge.upc.edu/web/people/albert-gil-moreno

|   |   |
|:--|:-:|
|  We gratefully acknowledge the support of [NVIDIA Corporation](http://www.nvidia.com/content/global/global.php) with the donation of the GeoForce GTX [Titan X](http://www.geforce.com/hardware/desktop-gpus/geforce-gtx-titan-x) used in this work. |  ![logo-nvidia] |
|  The Image ProcessingGroup at the UPC is a [SGR14 Consolidated Research Group](https://imatge.upc.edu/web/projects/sgr14-image-and-video-processing-group) recognized and sponsored by the Catalan Government (Generalitat de Catalunya) through its [AGAUR](http://agaur.gencat.cat/en/inici/index.html) office. |  ![logo-catalonia] |

[logo-nvidia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/nvidia.jpg "Logo of NVidia"
[logo-catalonia]: https://raw.githubusercontent.com/imatge-upc/saliency-2016-cvpr/master/logos/generalitat.jpg "Logo of Catalan government"

## Contact

If you have any general doubt about our work or code which may be of interest for other researchers, please use the [public issues section](https://github.com/marc-gorriz/KneeOA-CNNAttention/issues) on this github repo. Alternatively, drop us an e-mail at <mailto:xavier.giro@upc.edu>.
