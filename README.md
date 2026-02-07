# Pavement surface distress detection based on deep learning: A systematic review
A database including links to open-source datasets, codes, and papers analyzed in the review paper from 01/01/2020 to 12/31/2025. 

## Open-source datasets
- RDD
[Github](https://github.com/sekilab/RoadDamageDetector)
- CQU-BPDD 
[Github](https://github.com/DearCaat/Pavement-Distress-Classification)
- GAPs
[Project](https://www.tu-ilmenau.de/en/university/departments/department-of-computer-science-and-automation/profile/institutes-and-groups/institute-of-computer-and-systems-engineering/group-for-neuroinformatics-and-cognitive-robotics/data-sets-code/german-asphalt-pavement-distress-dataset-gaps)
- Crack500
[Github](https://github.com/fyangneil/pavement-crack-detection)
- DeepCrack
[Github](https://github.com/qinnzou/DeepCrack)
- Crack Forest 
[Github](https://github.com/cuilimeng/CrackForest-dataset)
- Mendeley Concrete crack dataset 
[Project](https://data.mendeley.com/datasets/5y9wdsg2zt/2)

## Label tools
- LabelImg
[Github](https://github.com/HumanSignal/labelImg)
- LabelMe
[Github](https://labelme.io/)
- Computer Vision Annotation Tool
[Project](https://www.cvat.ai/)

## Classical base models
- YOLO
[Github](https://github.com/ultralytics/ultralytics)
- U-Net
[Paper](https://arxiv.org/pdf/1505.04597)
[Project](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- Transformer
[Transformer](https://arxiv.org/pdf/1706.03762)
[Vision Transformer](https://arxiv.org/pdf/2010.11929)
[Detection Transformer](https://github.com/facebookresearch/detr)
[SegFormer](https://github.com/NVlabs/SegFormer)
[CLIP](https://arxiv.org/pdf/2103.00020)
- Segment Anything Models
[Github](https://github.com/facebookresearch/segment-anything)
- Generative Adversarial Nets
[GAN](https://arxiv.org/pdf/1406.2661)
[CGAN](https://arxiv.org/pdf/1411.1784)
[DCGAN](https://arxiv.org/pdf/1511.06434)
[SRGAN](https://arxiv.org/pdf/1609.04802)
[LSGAN](https://arxiv.org/pdf/1611.04076)
[Pix2Pix](https://arxiv.org/pdf/1611.07004)
[WGAN](https://arxiv.org/abs/1701.07875)
[CycleGAN](https://arxiv.org/pdf/1703.10593)
[BEGAN](https://arxiv.org/pdf/1703.10717)
[WGAN-GP](https://arxiv.org/pdf/1704.00028)
[ProGAN](https://arxiv.org/pdf/1710.10196)
[SAGAN](https://arxiv.org/pdf/1805.08318)
[BigGAN](https://arxiv.org/pdf/1809.11096)
[StyleGAN](https://arxiv.org/pdf/1812.04948)
- Diffusion Models
[DDPM](https://arxiv.org/pdf/2006.11239)
[DDIM](https://arxiv.org/pdf/2010.02502)
[DiT](https://arxiv.org/pdf/2212.09748)
[Stable Diffusion](https://arxiv.org/pdf/2112.10752)

## Model architectures
> Classified according to the model and arranged in chronological order.

### YOLO
> 2025
- Comparison of different deep learning algorithms in pavement crack detection
[Paper](https://www.tandfonline.com/doi/full/10.1080/14680629.2025.2498107)
- DMC-YOLOv11: an improved pavement damage detection model based on YOLOv11
[Paper](https://www.tandfonline.com/doi/full/10.1080/21642583.2025.2579987)
- PD-YOLO: A Multi-Scale Pavement Damage Detection Algorithm Based on Wavelet Transform and Global Attention
[Paper](https://ieeexplore.ieee.org/abstract/document/11161129)
- Research on pavement crack detection technology based on improved YOLO11
[Ppaer](https://ieeexplore.ieee.org/abstract/document/11149759)
- PotholeNet11:A YOLO11-Powered Real-Time Framework for Comprehensive Pothole Detection and Road Safety
[Paper](https://ieeexplore.ieee.org/abstract/document/11160097)
- Crack Detection Based on an Enhanced YOLO Method
[Paper](https://ieeexplore.ieee.org/abstract/document/11160593)
- An Enhanced YOLOv11-Based Approach for Pavement Distress Detection via Multi-Scal Feature Fusion and Adaptive Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/11137827?casa_token=VnYyw0BXbuwAAAAA:tdm8L-VIrPxqbexriw3uFwN_8zAoiJYg9Bhd1nmG7wj0DBm7xrccaWDsBPvOArR_UarUtQMOurk)

> 2024
- Multi-Object Detection for Daily Road Maintenance Inspection With UAV Based on Improved YOLOv8
[Paper](https://ieeexplore.ieee.org/abstract/document/10633267)
- Image Enhancement Method Utilizing YOLO Models to Recognize Road Markings at Night
[Paper](https://ieeexplore.ieee.org/abstract/document/10630821)
- A Novel Approach for Shadow Removel and Pavement Crack Detection Using Deep Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/10527997?casa_token=-SymeSwmDl0AAAAA:21JMS0u54WzY5QZjX2v1lXdc77i0sSEQZB2YCQSKmJPjmjyy7juphimeVB8KAtUepNTV_65x2Ds)
- Adaptive & Fine-Grained Domain Adaptation for Pavement Crack Segmentation using YOLOv8 Learning Framework
[Paper](https://ieeexplore.ieee.org/abstract/document/10533476)
- Deep Learning-Based Road Pavement Inspection by Integrating Visual Information and IMU
[Paper](https://doi.org/10.3390/info15040239)
- Automated Pavement Condition Index Assessment with Deep Learning and Image Analysis: An End-to-End Approach
[Paper](https://doi.org/10.3390/s24072333)
- A data-centric strategy to improve performance of automatic pavement defects detection
[Paper](https://doi.org/10.1016/j.autcon.2024.105334)
- Pavement crack instance segmentation using YOLOv7-WMF with connected feature fusion
[Paper](https://doi.org/10.1016/j.autcon.2024.105331)
- Automatic Detection of Pavement Marking Defects in Road Inspection Images Using Deep Learning
[Paper](https://doi.org/10.1061/JPCFEV.CFENG-4619)
- Automated detection and quantification of pavement cracking around manhole
[Paper](https://doi.org/10.1016/j.engappai.2023.107778)
- A Pavement Crack Detection Method via Deep Learning and a Binocular-Vision-Based Unmanned Aerial Vehicle
[Paper](https://doi.org/10.3390/app14051778)
- A Pavement Crack Detection and Evaluation Framework for a UAV Inspection System Based on Deep Learning
[Paper](https://doi.org/10.3390/app14031157)
- Automated Pavement Distress Detection and Classification Using YOLO-Based Deep Learning Models
[Paper](https://doi.org/10.15866/irece.v15i5.25213)
- Research and optimization of YOLO-based method for automatic pavement defect detection
[Paper](https://doi.org/10.3934/ERA.2024078)
- Detection of Road Cracks and Pavement Conditions Index Estimate Based on YOLOv5
[Paper](https://doi.org/10.1061/9780784485255.036)


> 2023
-

> 2022
- Empirical Study of the Performance of Object Detection Methods on Road Marking Dataset
[Paper](https://ieeexplore.ieee.org/abstract/document/10013909)
- The Application of a Pavement Distress Detection Method Based on FS-Net
[Paper](https://doi.org/10.3390/su14052715)
- Automatic Defect Detection of Pavement Diseases
[Paper](https://doi.org/10.3390/rs14194836)

> 2021
-

> 2020
- Smart City Lane Detection for Autonomous Vehicle
[Paper](https://ieeexplore.ieee.org/abstract/document/9251187)


### U-Net
> 2025
- 
- RepCrack: An efficient pavement crack segmentation method based on structural re-parameterization
[Paper](https://www.sciencedirect.com/science/article/pii/S095219762401950X)
- CrackMDM: Masked Modeling on DCT Domain for Efficient Pavement Crack Segmentation
[Paper](https://ieeexplore.ieee.org/abstract/document/11145882?casa_token=VmZjjRj0xY8AAAAA:HQ3glNwn0FwYqIe-mIFeE-VDJA9l7kf7etcrFpktkS_1fHVgfZnxje4NDox-qVNLWX4MXnjbLwk)
- Crack-MA: Automatic Pavement Crack Detection Based on Deep Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/11220466?casa_token=ygdlCv-KJWsAAAAA:VchXjLVQUMrd_oCPDKnBBlcn54eJe3OhdqsG1emQQZZIr7fDrrnrcIbYnoiU-uUEDfNG48P0Sas)
- Research on Road Damage Segmentation Algorithm based on Improved U-Shaped Network
[Paper](https://ieeexplore.ieee.org/abstract/document/10934684?casa_token=djjwe_k-3SsAAAAA:MzsWD8I_TmzJlstUwfEwJ1JxxIiNcaWazZlk0AormDz7T6jyrW2c72XBwVRQiQsQN6I5t9BMmIg)
- AsphaltCrackNet: A Novel Architecture for Classifying Cracks in Asphalt Pavement
[Paper](https://ieeexplore.ieee.org/abstract/document/11199027)

> 2024
- Encoder–decoder with pyramid region attention for pixel-level pavement crack recognition
[Paper](https://doi.org/10.1111/mice.13128)
- GGMNet: Pavement-Crack Detection Based on Global Context Awareness and Multi-Scale Fusion
[Paper](https://doi.org/10.3390/rs16101797)
- OUR-Net: A Multi-Frequency Network With Octave Max Unpooling and Octave Convolution Residual Block for Pavement Crack Segmentation
[Paper](https://ieeexplore.ieee.org/abstract/document/10542729)
- DepthCrackNet: A Deep Learning Model for Automatic Pavement Crack Detection
[Paper](https://www.mdpi.com/2313-433X/10/5/100)
- A multiscale enhanced pavement crack segmentation network coupling spectral and spatial information of UAV hyperspectral imagery
[Paper](https://doi.org/10.1016/j.jag.2024.103772)
- An attention-based progressive fusion network for pixelwise pavement crack detection
[Paper](https://doi.org/10.1016/j.measurement.2024.114159)
- Fine-grained damage detection of cement concrete pavement based on UAV remote sensing image segmentation and stitching
[Paper](https://doi.org/10.1016/j.measurement.2023.113844)
- The Effect of Visual Pavement Marking Properties on Deep Neural Network Detection Performance
[Paper](https://doi.org/10.1049/icp.2024.3323)
- UnderstAnding Bag of Tricks of Deep Learning-Based Semantic Segmentation in Pavement Crack Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10641520?casa_token=0MhbaGJ16-gAAAAA:MF140Q_UkiuCgSdWtFseaADEHIcsQ6XGkgloG8LoBd83-8RS_x6FG6slgyl3SNngCWOg6BWWVAU) 

> 2023
-

> 2022
- Automatic detection of deteriorated inverted-T patching using 3D laser imaging system based on a true story indiana
[Paper](https://doi.org/10.1093/iti/liac011)
- Pavement Crack Detection on BEV Based on Attention-Unet
[Paper](https://ieeexplore.ieee.org/abstract/document/9998932?casa_token=IbUqc3Va-XIAAAAA:EdnHDDBkJvmoCniYSmrZum_GPZZNYj0EkXTHnnruzEM41syFa3sq-WqOwQRikSIB8YG2h20Bc2w)

> 2021
- Road pavement crack detection using deep learning with synthetic data
[Paper](https://doi.org/10.1088/1757-899X/1019/1/012036)
- RiskIPN: Pavement Risk Database for Segmentation with Deep Learning
[Paper](https://doi.org/10.1007/978-3-030-89817-5_5)
- Key Points Estimation and Point Instance Segmentation Approach for Lane Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9460822)
- Automatic Crack Segmentation in Pavements using a Dilated Encoder-Decoder Network
[Paper](https://ieeexplore.ieee.org/abstract/document/9476956?casa_token=zj02LbgnxR4AAAAA:s4tz4lYcjLUjSy_v1eiQBekNdYNZ7b5vFlEd4-XCZTeVUXLDlDPQTq-A_tsPBlv9SR-yyytsozY)

> 2020
-


### Transformer
> 2025
- RTCNet: A novel real-time triple branch network for pavement crack semantic segmentation
[Paper](https://www.sciencedirect.com/science/article/pii/S1569843224007052)
[Dataset](https://github.com/NJSkate/BeijingHighway-dataset)
- Multi-Object Pavement Surface Feature Detection with CNN and Transformer Deep Learning Architecture
[Paper](https://ascelibrary.org/doi/abs/10.1061/9780784486214.032)
- ISTD-CrackNet: Hybrid CNN-transformer models focusing on fine-grained segmentation of multi-scale pavement cracks
[Paper](https://www.sciencedirect.com/science/article/pii/S0263224125005743)
- A transformer-based deep learning method for automatic pixel-level crack detection and feature quantification
[Paper](https://www.emerald.com/ecam/article/32/4/2455/1244805/A-transformer-based-deep-learning-method-for)
- A Hybrid Approach for Pavement Crack Detection Using Mask R-CNN and Vision Transformer Model
[Paper](https://www.sciencedirect.com/org/science/article/pii/S1546221825000086)
- Transformer–CNN Hybrid Framework for Pavement Pothole Segmentation
[Paper](https://www.mdpi.com/1424-8220/25/21/6756)
- A real-time multiscale pavement crack-detection model
[Paper](https://link.springer.com/article/10.1007/s11760-025-04836-8)
- BridgeFusionNet: A hybrid convolutional-transformer architecture for road surface crack
[Paper](https://www.sciencedirect.com/science/article/pii/S0952197625020937)
- Dual Framework for Fine Pavement Crack Detection Using Hybrid Attention
[Paper](https://ieeexplore.ieee.org/abstract/document/11085859?casa_token=K3WIZi3TPZ0AAAAA:QzsqyEU1T7UYhRhdnuN_gzw1rhnphunfn96-jer9DKzubvyFgsXFWbYJVH10ySf3Vh1UWFhYQrg)
- A Transformer-Based Pavement Crack Segmentation Model with Local Perception and Auxiliary Convolution Layers
[Paper](https://www.mdpi.com/2079-9292/14/14/2834)
- Autonomous Road Defects Segmentation Using Transformer-Based Deep Learning Models With Custom Dataset
[Paper](https://ieeexplore.ieee.org/abstract/document/11192307)
- Research on pavement crack detection technology based on deep learning
[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13792/137920H/Research-on-pavement-crack-detection-technology-based-on-deep-learning/10.1117/12.3090536.full)
- Deep learning approach for detailed block pavement distress segmentation
[Paper](https://www.tandfonline.com/doi/abs/10.1080/10298436.2025.2569616?casa_token=uTDxeSq2lN0AAAAA:05j--22r2Wsh4w6skTpGrnOjU5jXmluAjftwyS48WrRhHJO1kYf3Jc8kH9w8ztI-b4n-kUdO69dgdgk)

> 2024
- Automated pixel-level pavement marking detection based on a convolutional transformer
[Paper](https://doi.org/10.1016/j.engappai.2024.108416)
- A two-stage framework for pixel-level pavement surface crack detection
[Paper](https://doi.org/10.1016/j.engappai.2024.108312)
- Automated Multi-Type Pavement Distress Segmentation and Quantification Using Transformer Networks for Pavement Condition Index Prediction
[Paper](https://doi.org/10.3390/app14114709)
- Pavement Crack Detection Based on the Improved Swin-Unet Model
[Paper](https://doi.org/10.3390/buildings14051442)
- Bridge crack detection method based on multi-scale feature fusion
[Paper](https://dl.acm.org/doi/abs/10.1145/3677182.3677281)
- Dual-path network combining CNN and transformer for pavement crack segmentation
[Paper](https://doi.org/10.1016/j.autcon.2023.105217)
- SwinCrack: Pavement crack detection using convolutional swin-transformer network
[Paper](https://doi.org/10.1016/j.dsp.2023.104297)
- Detection of Pavement Cracks by Deep Learning Models of Transformer and UNet
[Paper](https://ieeexplore.ieee.org/abstract/document/10606398?casa_token=NURkGegh7vwAAAAA:r4UoecpMOgbIjqQJaPsRKMthr3VvXGqYNeuQnl9IsMeJVcRvWlFyceKVJAlALSrZ9Vyd-G6phk8)
- Research on Pavement Disease Detection Method Based on Improved Mask R-CNN
[Paper](https://ieeexplore.ieee.org/abstract/document/10729350?casa_token=Ia3VqhN1Ng4AAAAA:yZbXcbCxRMZhiqCtSHfugkmw9XJGaJ5kwzpvnQwh7dog3tuLB19C-uAEZWHbBd-EcQJDF1Aaca8)

> 2023
- Detection of Asphalt Pavement Cracks Based on Vision Transformer Improved YOLO V5
[Paper](https://doi.org/10.1061/JPEODX.PVENG-1180)
- Improving Highway Pavement Defect Detection via Swin Transformer Integrated TOOD Model
[Paper](https://ieeexplore.ieee.org/abstract/document/10733894?casa_token=KZWUmgm75nQAAAAA:hPCK68msFKn8b9pXwCXs8b900JRbiG_GgYVLsMjj-nvp5M4MCnsE6bm6S-ag7EWPJ0_oc2-gK1I)

> 2022
- Pavement crack detection from CCD images with a locally enhanced transformer network
[Paper](https://doi.org/10.1016/j.jag.2022.102825)

> 2021
-
> 2020
-

### Classical or novel CNN architectures
> 2025
- AI-based pothole detection: a comparative analysis using thermal and photographic imaging
[Paper](https://www.tandfonline.com/doi/abs/10.1080/10298436.2025.2559863?casa_token=RC4PEshOKjkAAAAA:D6IXbE9NQXcvYag_PZzvLtxygSjKt5iP4DAevi-VXIj4bJsil-hBMORuV2evHqPyD-b4nLnl9X_Gnws)
- Enhancing Anomaly Detection in Cycling Paths Using a Hybrid LSTM-VQ-VAE Deep Learning Model
[Paper](https://dl.acm.org/doi/full/10.1145/3719384.3719464)
> 2024
- Pixel-level detection of multiple pavement distresses and surface design features with ShuttleNetV2
[Paper](https://doi.org/10.1177/14759217231183656)
- Research on pothole road recognition based on CAMB attention mechanism and Mish function improved convolutional neural network MobileNetV3
[Paper](https://doi.org/10.1145/3673277.3673359)
- Advancement in Pavement Condition Assessment: an AI-Based Computer Vision Approach
[Paper](https://ieeexplore.ieee.org/abstract/document/10776158?casa_token=7SatGO_lhusAAAAA:-YE7uF8lgZZgs77zYQsyzR1PxQtO6gJuKwciDSfv2axMGOLGyN_VpP7pChH2IstF4MYWK1ekkC0)
- CNN-based network with multi-scale context feature and attention mechanism for automatic pavement crack segmentation
[Paper](https://www.sciencedirect.com/science/article/pii/S0926580524002188?casa_token=WievfcMJtCgAAAAA:Abpu45vucm0ShW7JZqnS-erbcDrH8yRgED6RFQve7ioEzOw3xiNPNjNBWBh1F2mLYXHMNEf0vNTu)
- An Automated Instance Segmentation Method for Crack Detection Integrated with CrackMover Data Augmentation
[Paper](https://doi.org/10.3390/s24020446)
> 2023
- Learning Light Fields for Improved Lane Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9999224)

> 2022
- An Iteratively Optimized Patch Label Inference Network for Automatic Pavement Distress Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9447759)
[Github](https://github.com/DearCaat/ioplin)
[Dataset](https://dearcaat.github.io/CQU-BPDD/)
- Crack Detection and Comparison Study Based on Faster R-CNN and Mask R-CNN
[Paper](https://doi.org/10.3390/s22031215)
- Deep learning and infrared thermography for asphalt pavement crack severity classification
[Paper](https://doi.org/10.1016/j.autcon.2022.104383)
- Deep Metric Learning-Based for Multi-Target Few-Shot Pavement Distress Classification
[Paper](https://ieeexplore.ieee.org/abstract/document/9459530?casa_token=ad6-6X-zSaoAAAAA:SJVx1FRXRB2cxx6txgHJsPcS74oxNhmVx7ePObAsG-7-EeZdifNlZY564KjrmytPIinBj2BphDU)
> 2021
- Multi-stage Deep Learning Technique with a Cascaded Classifier for Turn Lanes Recognition
[Paper](https://ieeexplore.ieee.org/abstract/document/9659973)
> 2020
-

### GAN
> 2025
- An Innovative Pavement Performance Prediction Method Based on Few-Shot Learning
[Paper](https://ascelibrary.org/doi/abs/10.1061/JPEODX.PVENG-1583?casa_token=a3VzmOXJPxsAAAAA:R_PO0m1oKMDChQzAsYCd98uTQ0D1s_q4oEql4solbM1dHUeKBk2M1KMNUVvB6b03ZGEaj2mc936J)
- Research on the automatic crack detection system of asphalt pavement based on deep learning
[Paper](https://dl.acm.org/doi/full/10.1145/3701100.3701125)
- Conditional Pavement Crack Data Generation for Selective Data Augmentation Using GANs
[Paper](https://ieeexplore.ieee.org/abstract/document/11144507?casa_token=w5Mx4Cg7ZDgAAAAA:9mL6jxODPMAl8eArcDQn3gOla-MvhgAotUqGdScBmS0fmzOsJ2nFCUJcDfC4t0M4VWtO0wjvXo8)
- Data generation for asphalt pavement evaluation: Deep learning-based insights from generative models
[Paper](https://www.sciencedirect.com/science/article/pii/S2214509525009143)
- High quality training set development for road crack detection using progressive data-driven models with integrated identification
[Paper](https://www.sciencedirect.com/science/article/pii/S2214509525012148)
- Enhancing structural health monitoring: Conditional GAN-based crack detection in concrete and asphalt surfaces
[Paper](https://journals.sagepub.com/doi/full/10.1177/13694332251381215?casa_token=iGxe894Bi1gAAAAA%3ALV4cUdmN2LvrR7c1YufSsE5bZBgLWUjyMMZ2-3oKcjo_U3NFU7ShFda8KdCCN6-7R-53Eo-9pEX634Q)
- Generative adversarial network for real-time identification and pixel-level annotation of highway pavement distresses
[Paper](https://www.sciencedirect.com/science/article/pii/S0926580525001621?casa_token=ldgqOxCb_pUAAAAA:ZrNZwlVGZoZ3dNf_hTa_l6X589MOYI6NKawJ-opdsmTh5gg6Mks-xi8PNkTQxKHCZ8WpXmsEf7JH)
- A Shadow-Robust Pavement Damage Detection Framework Based on RACycle-GAN and DDE-YOLOv8
[Paper](https://ieeexplore.ieee.org/abstract/document/10965794?casa_token=MYgrJESZupsAAAAA:HR1RkZWVqv5WRniDDroxXykfNdFUlhuD9iF8xU7rzltWVu7xcp575Xb1q1ZBDqY51IFbn7Oegk4)
> 2024
- A three-stage pavement image crack detection framework with positive sample augmentation
[Paper](https://doi.org/10.1016/j.engappai.2023.107624)
- Bridging Data Distribution Gaps: Test-Time Adaptation for Enhancing Cross-Scenario Pavement Distress Detection
[Paper](https://openurl.ebsco.com/EPDB%3Agcd%3A16%3A24966864/detailv2?sid=ebsco%3Aplink%3Ascholar&id=ebsco%3Agcd%3A181961449&crl=c&link_origin=scholar.google.com)
- Leveraging a deep learning generative model to enhance recognition of minor asphalt defects
[Paper](https://www.nature.com/articles/s41598-024-80199-3)
- Classification of pothole pavement based on pseudo-sample generation augmentation
[Paper](https://ieeexplore.ieee.org/abstract/document/10845491)
- CrackCLF: Automatic Pavement Crack Detection Based on Closed-Loop Feedback
[Paper](https://ieeexplore.ieee.org/abstract/document/10328544?casa_token=n7GixT7X1BoAAAAA:P1Ocl6NNnJNgJVqMZQkWZvf3zoFo1v0o5mTqsBmQUORuy5HT-Exc2xm4GW06VXWQ6UdbtHjMZ7o)
- A pavement crack synthesis method based on conditional generative adversarial networks
[Paper](https://doi.org/10.3934/mbe.2024038)
- Research on pavement crack detection algorithm based on Adversarial and Depth-guided Network
[Paper](https://doi.org/10.1117/12.3021078)
> 2023
- Integrated APC-GAN and AttuNet Framework for Automated Pavement Crack Pixel-Level Segmentation: A New Solution to Small Training Datasets
[Paper](https://ieeexplore.ieee.org/abstract/document/10021216?casa_token=qWIzY4gMcE8AAAAA:q7EwdRDcaGbUGdMdyapkShvj02khaorwPpFKMjSQ9niZzbDvBD40-ii7g4n-iBiHOt6M983msMY)
- Pavement Image Enhancement in Pixel-Wise Based on Multi-Level Semantic Information
[Paper](https://ieeexplore.ieee.org/abstract/document/10220435?casa_token=7MHMRoPhyggAAAAA:ZiMss2ecBKtGpXajA-uZPu-qxpOG0W9eOV1NhfbULkzGW90csGI-62bsSsPpLLUFia78NbjAZ30)
- A deeper generative adversarial network for grooved cement concrete pavement crack detection
[Paper](https://doi.org/10.1016/j.engappai.2022.105808)
- Enhancing Pavement Distress Detection Using a Morphological Constraints-Based Data Augmentation Method
[Paper](https://doi.org/10.3390/coatings13040764)
> 2022
- Data Augmentation by an Additional Self-Supervised CycleGAN-Based for Shadowed Pavement Detection
[Paper](https://www.mdpi.com/2071-1050/14/21/14304)
- Pavement crack detection algorithm based on generative adversarial network and convolutional neural network under small samples
[Paper](https://doi.org/10.1016/j.measurement.2022.111219)
- Lane Detection based on Object Detection and Image-to-image Translation
[Paper](https://ieeexplore.ieee.org/abstract/document/9412400?casa_token=a8V-rE0EU-oAAAAA:-3hO5v7wv5aJY3Jaj2z1lIJwQip6bs71YAlTdkcz4TlbtbNikyVCQrqWTV9_4cDIpHIQktP8OH4)
- LanePainter: Lane Marks Enhancement via Generative Adversarial Network
[Paper](https://ieeexplore.ieee.org/abstract/document/9956446?casa_token=C2uVH-eOMJYAAAAA:awf9PYoawEhPMcDmkDWdvMk1rsMD1GKE6IzL-4jPihT0OhQwWj8GWPZXn1pBAXDg-K8lFz_38LM)
- Automatic Detection and Counting System for Pavement Cracks Based on PCGAN and YOLO-MF
[Paper](https://ieeexplore.ieee.org/abstract/document/9745793?casa_token=gcfiplorQGAAAAAA:RYLY2v43kATuKGafbWFhn18Bg5I5BaXupivc2YG6JJu6Xh28QtHZQ9j2zjg_Sg5CUO-kbjAiCIw)
- Super-Resolution Reconstruction Method of Pavement Crack Images Based on an Improved Generative Adversarial Network
[Paper](https://doi.org/10.3390/s22239092)
> 2021
- Virtual generation of pavement crack images based on improved deep convolutional generative adversarial network
[Paper](https://doi.org/10.1016/j.engappai.2021.104376)
> 2020
- CrackGAN: Pavement Crack Detection Using Partially Accurate Ground Truths Based on Generative Adversarial Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/9089245?casa_token=8IAQWpEf6cwAAAAA:nDhA3FD09s-tjaIS8axC5A6d3wx0v1n3VLXL-CSP0Jv3XrFz1-zveur_NJQpbmkCAQyXezkqIRU)
- A cost effective solution for pavement crack inspection using cameras and deep neural networks
[Paper](https://doi.org/10.1016/j.conbuildmat.2020.119397)
- Pavement crack detection and recognition using the architecture of segNet
[Paper](https://doi.org/10.1016/j.jii.2020.100144)
- Guided Dual Network Based Transfer with an Embedded Loss for Lane Detection in Nighttime Scene
[Paper](https://ieeexplore.ieee.org/abstract/document/9345241?casa_token=czaCl-6sIKcAAAAA:l-XgxAsxoEz36s_-cI59etp03Thr9wNScUiHkiyhxTGK9rcyKYCkYpWTaaUmXtg58RWiAuqx7O4)

### Diffusion models
> 2024
- A controllable generative model for generating pavement crack images in complex scenes
[Paper](https://doi.org/10.1111/mice.13171)
- CrackDiffusion: A two-stage semantic segmentation framework for pavement crack combining unsupervised and supervised processes
[Paper](https://doi.org/10.1016/j.autcon.2024.105332)
- Improving detection of asphalt distresses with deep learning-based diffusion model for intelligent road maintenance
[Paper](https://doi.org/10.1016/j.dibe.2023.100315)
- Enhancing pavement crack segmentation via semantic diffusion synthesis model for strategic road assessment
[Paper](https://www.sciencedirect.com/science/article/pii/S2590123024010004)

### 3D models
> 2025
- On the Applicability of Multimodal Neural Network Methods for Determining the Quality of the Road Surface
[Paper](https://ieeexplore.ieee.org/abstract/document/11079337?casa_token=EFb8zxRYWM4AAAAA:qYWdOld1h8v1max2YARbdI2Bod4lGQ_gQKe0O2l5GAVvkq7oGC5qR9uZdZzlYMNhi8P5vsuqWE0)
- Meso-structural characterization of asphalt mixture via RAN-UNet segmentation and 3D reconstruction: Insights into FAM void evolution
[Paper](https://www.sciencedirect.com/science/article/pii/S0950061825040462)
- Aggregate-level 3D analysis of asphalt pavement deterioration using laser scanning and vision transformer
[Paper](https://www.mdpi.com/1424-8220/25/21/6756)
- On the Applicability of Multimodal Neural Network Methods for Determining the Quality of the Road Surface
[Paper](https://ieeexplore.ieee.org/abstract/document/11079337)
- Comparative Analysis of Upsampling Algorithms for Pavement Point Clouds: Toward 3D Pavement Distress Inspection
[Paper](https://ascelibrary.org/doi/abs/10.1061/JPEODX.PVENG-1830?casa_token=gDort96YlAsAAAAA:JjKCthdyW5VjNylLAKhCn7L84lcP2Oxd0rNs7NbgHUgJJSj7XLrkeSztNH3tIBr0y9GmK3SQ5V7-)
> 2024
- LiDAR-Based Automatic Pavement Distress Detection and Management Using Deep Learning and BIM
[Paper](https://doi.org/10.1061/JCEMD4.COENG-14358)
- PAN: Improved PointNet++ for Pavement Crack Information Extraction
[Paper](https://www.mdpi.com/2079-9292/13/16/3340)
- Pavement Point Cloud Upsampling Based on Transformer: Toward Enhancing 3D Pavement Data
[Paper](https://ieeexplore.ieee.org/abstract/document/10682468?casa_token=QfXCI5DahRsAAAAA:7IRRhJpQ20DU6ngatE8efjGZVZgLL1fiUnjnXJiJ-Z3AF7CAqqhhXux0YTK8Z1aVNYwYrpHW1n0)
> 2023
- Pavement Crack Detection Based on 3D Edge Representation and Data Communication With Digital Twins
[Paper](https://ieeexplore.ieee.org/abstract/document/9852817?casa_token=z5Au8Y8KJ08AAAAA:mXlOwSYVGWVpzH_2IKMbeD0jl55i0F0EkZrKpZozmI3yyqCBcWPN-t8_AIkEeTDv7yG2arzMw7Q)

### PINN
> 2025
- Pavement Cracking Prediction Models Based on Deep Learning Physics-Informed Neural Network
[Paper](https://journals.sagepub.com/doi/full/10.1177/03611981251372087)

### Mamba
- GLoU-MiT: Lightweight Global-Local Mamba-Guided U-mix transformer for UAV-based pavement crack segmentation
[Paper](https://www.sciencedirect.com/science/article/pii/S1474034625002770?casa_token=o3HeAVUzCzAAAAAA:uiizuq2xIvDKDhtDrHaArKfg5DrMzqRR3KqaKTlV-EhZrWBVAUJ4Ym82bacG1CrN5S6cZgeSiLYz)

### GCN
> 2025
- Two-Branch Multiscale Context with Multi-View Spatial–Temporal Graph Convolutional Networks for Pavement Fatigue Cracking Prediction
[Paper](https://link.springer.com/article/10.1007/s11668-025-02291-8)
- Hybrid graph convolutional and deep convolutional networks for enhanced pavement crack detection
[Paper](https://www.sciencedirect.com/science/article/pii/S0952197625002271?casa_token=5ggeFSNDQBQAAAAA:M_rFlU63kF9TPqE6F-lhij58Tx4fOabb6ie7s1uj5YWLMbSzeEYuQ9aDJhnPwvz3TBW7V7_bySgF)
> 2022
- GCN-Based Pavement Crack Detection Using Mobile LiDAR Point Clouds
[Paper](https://ieeexplore.ieee.org/abstract/document/9508901)
- The Improvement of Automated Crack Segmentation on Concrete Pavement with Graph Network
[Paper](https://doi.org/10.1155/2022/2238095)

## Learning paradigms
### Transfer learning
> Foundation models
>> 2025
- PaveSAM–segment anything for pavement distress
[Paper](https://www.tandfonline.com/doi/full/10.1080/14680629.2024.2374863)
- Semi-supervised crack detection using segment anything model and deep transfer learning
[Paper](https://www.sciencedirect.com/science/article/pii/S0926580524006356)
- CLIP-Guided Cross-Modal Feature Fusion based Few-Shot Learning for Nighttime Pavement Defect Detection
[Paper](https://www.ieice.org/publications/proceedings/summary.php?iconf=MVA&session_num=O2-2&number=O2-2-2&year=2025)
- Evaluating Prompt-Guided Vision-Language Models for Crack Segmentation
[Paper](https://ieeexplore.ieee.org/abstract/document/11244601)
- Enhanced Crack Segmentation Using Meta’s Segment Anything Model with Low-Cost Ground Truths and Multimodal Prompts
[Paper](https://journals.sagepub.com/doi/full/10.1177/03611981251322484)
- Fine-tuning large vision model for multimodal fusion in asphalt pavement crack segmentation
[Paper](https://www.tandfonline.com/doi/full/10.1080/10298436.2025.2526158)
- An Integrated Framework with SAM and OCR for Pavement Crack Quantification and Geospatial Mapping
[Paper](https://www.mdpi.com/2412-3811/10/12/348)
> Classical models
>> 2025
- Interpretation and understanding of asphalt crack detection deep learning models using integrated gradient (I.G.) maps
[Paper](https://www.sciencedirect.com/science/article/pii/S2214509525004115)
- A Multi-Scale and CNN-Based Approach for Robust Crack Detection in Noisy Environments
[Paper](https://ieeexplore.ieee.org/abstract/document/11013365?casa_token=hxLHs9Wnl5cAAAAA:QPABQImUgZOCEtoxGjZUdOxjdOKXSpZa5o35s9JqJu4CfFNHwTxl8KmOeXRW7gSeUd6Q7hrz6Pk)
- Architecture for pavement pothole evaluation using deep learning, machine vision, and fuzzy logic
[Paper](https://www.sciencedirect.com/science/article/pii/S2214509525002384)
- Evaluation of RGB-D Image for Counting Exposed Aggregate Number on Pavement Surface Based on Computer Vision Technique
[Paper](https://link.springer.com/article/10.1007/s10921-024-01144-y)
- Nondestructive Detection of Road Defects Using YOLOv9 Neural Network and Transfer Learning
[Paper](https://link.springer.com/article/10.3103/S8756699025000010)
- Integrating Transfer Learning and U-Net for Accurate Detection and Segmentation of Asphalt Pavement Bleeding
[Paper](https://journals.sagepub.com/doi/full/10.1177/03611981251348446?casa_token=IYxmU6JHrK0AAAAA%3Ao6ce8YS3dbwUtLFIGV3XI10ej-Tqvvvk0yY9D5G5HS5IZITGkw_PURZp-8dTgzSyZJYRztEt8klLZiQ)
- Asphalt Pavement Distress Detection by Transfer Learning with Multi-head Attention Technique
[Paper](https://hrcak.srce.hr/327832)
- Modified MobileNetV2 transfer learning model to detect road potholes
[Paper](https://peerj.com/articles/cs-2519/)
- GAF-Net: A new automated segmentation method based on multiscale feature fusion and feedback module
[Paper](https://www.sciencedirect.com/science/article/pii/S0167865524003386?casa_token=kfAuknB2x3AAAAAA:tkBKvfM8l3XkT1tfNIjZ_NSUZ98LmgCgXUOrQwZqjH9FwxN_ABZaOHlzYdhvNWVSxOvfdPKuHcSx)
- Enhancing asphalt mix design with transfer learning and hybrid artificial intelligence
[Paper](https://www.sciencedirect.com/science/article/pii/S2214509525010563)
- Image-Based Pavement Pothole Detection System Using Deep Learning
[Paper](https://link.springer.com/chapter/10.1007/978-981-96-3725-6_26)
- Asphalt pavement deformation distress evaluation using deep learning
[Paper](https://www.tandfonline.com/doi/abs/10.1080/10298436.2025.2524738?casa_token=1tF4AFcz3xwAAAAA:iU_kRfAHK-s7cphpwiVqjbjO2IC1nUnsENwvn4a5JZZG36UhF92tY87h60Xloj_xK5o9EftlPIhvLkM)
- Smart Roadway Monitoring: Pothole Detection and Mapping via Google Street View
[Paper](https://link.springer.com/chapter/10.1007/978-3-031-85923-6_12)
>> 2024
- GoogleNet transfer learning with improved gorilla optimized kernel extreme learning machine for accurate detection of asphalt pavement cracks
[Paper](https://journals.sagepub.com/doi/full/10.1177/14759217231215419?casa_token=25WeP1bbInMAAAAA%3AglRR23e-R2gucPftO_LuyKoLBMauD5um0ceIF6WbH10foEEVKr-3f86LK6vjsJCOd-QCeuw2Qrhx3es)
- Evaluation and optimisation of pre-trained CNN models for asphalt pavement crack detection and classification
[Paper](https://doi.org/10.1016/j.autcon.2024.105297)
- Study of Semantic Segmentation Models for the Detection of Pavement Degradation Using Deep Convolutional Neural Networks
[Paper](https://doi.org/10.1007/978-3-031-75329-9_19)
- AUTOMATIC CRACK CLASSIFICATION ON ASPHALT PAVEMENT SURFACES USING CONVOLUTIONAL NEURAL NETWORKS AND TRANSFER LEARNING
[Paper](https://doi.org/10.36680/j.itcon.2024.055)
- Pothole Detection of Road Pavement by Modified MobileNetV2 for Transfer Learning
[Paper](https://doi.org/10.1007/978-981-97-3180-0_34)
- Pothole Road Detection and Recognition Based on Deep Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/10743926?casa_token=A6thUI9ZJNYAAAAA:ZNLGhYFzCD_5V1NupMbRk8h8erMCRauEWdiRFTJl0A6IHrppST8yWVzTpO6DJpkM5nW103Uj6D0)
- Harnessing Deep Learning Techniques for Enhanced Detection and Classification of Cracks in Pavement Imagery
[Paper](https://doi.org/10.1016/j.procs.2024.05.045)
- Yolo and RetinaNet Ensemble Transfer Learning Detector: Application in Pavement Distress
[Paper](https://doi.org/10.1007/978-3-031-56998-2_3)
- Multilabel CNN Model for Asphalt Distress Classification
[Paper](https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-5500)
- Deep Learning Approaches for Road Damage Detection Using YOLOv8 and Custom Convolutional Neural Network
[Paper](https://ieeexplore.ieee.org/abstract/document/10913309)
- Revolutionizing Road Safety: AI-Powered Road Defect Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10486759)
- Classification of Asphalt Pavement Defects for Sustainable Road Development Using a Novel Hybrid Technology Based on Clustering Deep Features
[Paper](https://www.mdpi.com/2071-1050/16/22/10145)
>> 2023
- A Detection and Classification Method of Asphalt Pavement Crack based on Vision Transformer
[Paper](https://doi.org/10.1145/3656766.3656970)
- Automated classification and detection of multiple pavement distress images based on deep learning
[Paper](https://doi.org/10.1016/j.jtte.2021.04.008)
- Preprocessing of Crack Recognition: Automatic Crack-Location Method Based on Deep Learning
[Paper](https://doi.org/10.1061/(ASCE)MT.1943-5533.0004605)
- A Deep Learning-Based Object Detection Framework for Automatic Asphalt Pavement Patch Detection Using Laser Profiling Images
[Paper](https://doi.org/10.1007/978-3-031-44137-0_18)
- Pavement Transverse Cracking Detection Based on the Vehicle Vertical Acceleration Performed Via Transfer Learning and Wavelet Scattering Transform
[Paper](https://ieeexplore.ieee.org/abstract/document/10332812?casa_token=361kWAx1AVQAAAAA:MzLxWy_EPYf86FfQg6QsQzTr7dhosBbZfHYbGuTTB7H5h902SYgO7GIArVKvEfcl2AWouqoqfQY)
>> 2022
- Transfer learning based deep convolutional neural network model for pavement crack detection from images
[Paper](https://doi.org/10.22075/ijnaa.2021.24521.2762)
- Crack Detection and Classification in Moroccan Pavement Using Convolutional Neural Network
[Paper](https://doi.org/10.3390/infrastructures7110152)
- Optimization and sensitivity analysis of existing deep learning models for pavement surface monitoring using low-quality images
[Paper](https://doi.org/10.1016/j.autcon.2022.104332)
- Road Marking Damage Detection Based on Deep Learning for Infrastructure Evaluation in Emerging Autonomous Driving
[Paper](https://ieeexplore.ieee.org/abstract/document/9843894?casa_token=9_lxH5ud8ZkAAAAA:M9mEv7F0Dnu5Lz1WcdQ0eoMS1ukd_Zzk9I-GdKxmsj3xK6yuR6AOsc7BVnPlpyDXNnukpPJFzkI)
- Automatic crack detection in the pavement with lion optimization algorithm using deep learning techniques
[Paper](https://doi.org/10.1007/s00170-022-10724-z)
- Deep learning-based thermal image analysis for pavement defect detection and classification considering complex pavement conditions
[Paper](https://doi.org/10.3390/rs14010106)
- Asphalt Pavement Crack Image Screening by Transformer-Based Model
[Paper](https://ieeexplore.ieee.org/abstract/document/9853351?casa_token=kuncTjJtyL8AAAAA:Phnsc9MKpG90VKeT1iyNthSNH7tIcNlpn621CBt2WOjFrQujE2ju9f1-d8Wj9XFrS1YGc9NssGU)
- Transfer Learning-based Ensemble Deep Learning for Road Cracks Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9931581?casa_token=1o2UaaVPPhsAAAAA:fTx03jbwIT4Ot4lLpuLmAKOaqmPdkqlKxjiNdHK_tybSniAlgALaeQv5k-7JgRfQaCfQUJ9FLkI)
>> 2021
- Deep Learning-Based Pothole Detection System with Aerial Image
[Paper](https://ieeexplore.ieee.org/abstract/document/10487534)
- A State-of-the-Art Survey of Transfer Learning in Structural Health Monitoring
[Paper](https://ieeexplore.ieee.org/abstract/document/9664171?casa_token=yFUarIfEMqgAAAAA:L9T85YYseirSKSuRCgoeD9Geet9QaPwPTHY-zlBIaKtnot_c74iedA2wrIpfosdRDYixBT4XFA4)
- Evolving Pre-Trained CNN Using Two-Layers Optimizer for Road Damage Detection From Drone Images
[Paper](https://ieeexplore.ieee.org/abstract/document/9627890)
>> 2020
- Scale–Space Data Augmentation for Deep Transfer Learning of Crack Damage from Small Sized Datasets
[Paper](https://doi.org/10.1007/s10921-020-00715-z)
- An Efficient and Scalable Deep Learning Approach for Road Damage Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9377751?casa_token=WcR8LghC93EAAAAA:fLG3m-UD6kaIDpC2LvweCjVLReDUzcgOKATEBb9KPwrKs0hhKQ--jmCjBHInp7ToSmix1hiceZ4)

### Semi-supervised
> 2025
- Comparison of Deep Learning Methods and a Transfer-Learning Semisupervised Generative-Adversarial-Network Combined Framework for Pavement Crack Image Identification
[Paper](https://link.springer.com/article/10.1134/S1054661824701426)
> 2023
- A Semi-Supervised Learning Approach for Pixel-Level Pavement Anomaly Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10106612?casa_token=CPvPnGkD3DQAAAAA:VUer6aGRJ0_qxUNXjko-2E0tn0uAvLJWiUTfrF9JXmhKZFwrCScCewtBpJkuGAfSIqPXFBFFwoA)
> 2020
- Semi-Supervised Semantic Segmentation Using Adversarial Learning for Pavement Crack Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9032091)

### Weakly-supervised
> 2025
- A weakly-supervised deep learning model for end-to-end detection of airfield pavement distress
[Paper](https://www.sciencedirect.com/science/article/pii/S2046043024000303)
> 2024
- A weakly-supervised transformer-based hybrid network with multi-attention for pavement crack detection
[Paper](https://www.sciencedirect.com/science/article/pii/S0950061823038527?via%3Dihub)
- A Pavement Crack Translator for Data Augmentation and Pixel-Level Detection Based on Weakly Supervised Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/10557475?casa_token=HjQMyZYJdygAAAAA:SGIyjGXU5uth2GAsIYHqI5VRZh9DadOtI8DcaBi5rJne4_PR9RWrFt8sgbLxghn4s-M-iudETJY)
> 2023
- Weakly Supervised Patch Label Inference Networks for Efficient Pavement Distress Detection and Recognition in the Wild
[Paper](https://ieeexplore.ieee.org/abstract/document/10050387)
[Github](https://github.com/DearCaat/wsplin)
- Weakly supervised pavement crack semantic segmentation based on multi-scale object localization and incremental annotation refinement
[Paper](https://link.springer.com/article/10.1007/s10489-022-04212-w)
> 2022
- Weakly supervised convolutional neural network for pavement crack segmentation
[Paper](https://academic.oup.com/iti/article/doi/10.1093/iti/liac013/6798398)
- Investigation of pavement crack detection based on deep learning method using weakly supervised instance segmentation framework
[Paper](https://www.sciencedirect.com/science/article/pii/S0950061822027726?via%3Dihub)
> 2020
- Weakly supervised network based intelligent identification of cracks in asphalt concrete bridge deck
[Paper](https://www.sciencedirect.com/science/article/pii/S1110016820300910)

### Unsupervised/Self-supervised
> 2025
- Efficient crack segmentation with multi-decoder networks and enhanced feature fusion
[Paper](https://www.sciencedirect.com/science/article/pii/S0952197625006979)
[Github](https://github.com/AmmarOkran/CrackMaster)
- A Generative Approach to Generalize Deep Learning Models for Pavement Distress Segmentation
[Paper](https://link.springer.com/article/10.1007/s42421-025-00118-4)
- CL-PSDD: Contrastive Learning for Adaptive Generalized Pavement Surface Distress Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10843152)
- Unsupervised pavement rutting detection using structured light and area-based deep learning
[Paper](https://www.sciencedirect.com/science/article/pii/S0926580525002754)
- Pavement Crack Segmentation Based on Synthetic Data Sets and Unsupervised Domain Adaptation
[Paper](https://ascelibrary.org/doi/abs/10.1061/JCCEE5.CPENG-6675?casa_token=YgrE5gGsKCgAAAAA:KAOZps4evn-qHJxejMqVKJIZ9xJ86NMmHfEmy9CkU4MnN7TfdO8HI7eMqKnEK2lK9gB9g_k-H_py)
- A Pavement Crack Registration and Change Identification Method Based on Unsupervised Deep Neural Network
[Paper](https://ieeexplore.ieee.org/abstract/document/10766352)
- Photogrammetry-Driven Detection of Structural Peeling in Joints and Corners of Rigid Pavements Using an Unsupervised Learning Meta-Architecture
[Paper](https://ieeexplore.ieee.org/abstract/document/10918994/)
- The remote sensing method for large-scale asphalt pavement aging assessment with automated sample generation and deep learning
[Paper](https://www.nature.com/articles/s41598-025-29966-4)
- A Spatial Contexts-Informed Self-Supervised Learning Approach for Pavement Distress Segmentation
[Paper](https://ieeexplore.ieee.org/abstract/document/11192102?casa_token=FkmFN_yX1UQAAAAA:sWilsM-IiRQCoK_13PLZe9y2TqshPOiuFEjaV9DCIUlKT7M4FXwvj8M4e6SJaGviQ0diY1CE5kQ)
> 2024
- A Self-Supervised Learning Approach to Road Anomaly Detection Using Masked Autoencoders
[Paper](https://ascelibrary.org/doi/10.1061/9780784485538.047)
- An Unsupervised Learning Approach for Pavement Distress Diagnosis via Siamese Networks
[Paper](https://ieeexplore.ieee.org/abstract/document/10769539)
- Two-stage framework with improved U-Net based on self-supervised contrastive learning for pavement crack segmentation
[Paper](https://doi.org/10.1016/j.eswa.2023.122406)
> 2022
- Pavement anomaly detection based on transformer and self-supervised learning
[Paper](https://doi.org/10.1016/j.autcon.2022.104544)
> 2021
- Memory-augment convolutional Autoencoder for unsupervised pavement crack classification
[Paper](https://ieeexplore.ieee.org/abstract/document/9727812/)
- Unsupervised Deep Learning for Road Crack Classification by Fusing Convolutional Neural Network and K_Means Clustering
[Paper](https://ascelibrary.org/doi/10.1061/JPEODX.0000322)
> 2020
- Unsupervised Pixel-level Crack Detection Based on Generative Adversarial Network
[Paper](https://doi.org/10.1145/3404716.3404720)

### Reinforce learning
> 2024
- Pavement Crack Detection Algorithm Based on Reinforcement Learning and Traditional Supervised Learning
[Paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/13089/130890U/Pavement-crack-detection-algorithm-based-on-reinforcement-learning-and-traditional/10.1117/12.3021180.full)
> 2021
- Intelligent preprocessing selection for pavement crack detection based on deep reinforcement learning
[Paper](https://doi.org/10.18293/SEKE2021-062)

### Ensemble learning
> 2025
- Enhancing Marshall stability of asphalt concrete using a hybrid deep neural network and ensemble learning
[Paper](https://www.sciencedirect.com/science/article/pii/S221450952500960X)

## Inference acceleration
> 2025
- YOLOv11n-MG: A Road Disease Detection Algorithm Based on Lightweight Feature Extraction and Global Attention Mechanism
[Paper](https://ebooks.iospress.nl/doi/10.3233/ATDE250776)
- An ultra-lightweight network combining Mamba and frequency-domain feature extraction for pavement tiny-crack segmentation
[Paper](https://www.sciencedirect.com/science/article/pii/S0957417424028082?casa_token=v0qmkNBXKqcAAAAA:ncO-cxeePAlIdRlJba9gqtnDScFQlKaV_7U63v-sQ2I_0IIBoEp79D7rM0CiRlV-Sq6Tp1yvpI3z)
- LMD_YOLO: A Lightweight and Efficient Model for Pavement Defects Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10949187)
- Real-Time Road Damage Detection Using an Optimized YOLOv9s-Fusion in IoT Infrastructure
[Paper](https://ieeexplore.ieee.org/abstract/document/10883340?casa_token=S18wx9iAfCsAAAAA:RjFopFRPzeOzgCyN1KpMqwRRx_fY5Rfq22v0Cpi87fCpHoV-Rpuoieqdid1RW-k5UjJ-dPoKpmA)
- LPDDN: An Embedded Real-Time Pavement Defect Detection Model Under Computationally Constrained Conditions
[Paper](https://link.springer.com/chapter/10.1007/978-3-032-03632-2_2)
- L-YOLO-HR: implementing lightweight and efficient pavement distress detection by enhancement of the spatial information extractions of high-resolution features
[Paper](https://iopscience.iop.org/article/10.1088/2631-8695/ae0f01/meta)
- YOLOv8n-SBP: A Lightweight and Efficient Model for Pavement Distress Detection
[Paper](https://ascelibrary.org/doi/full/10.1061/JPEODX.PVENG-1815)
-MDCCM: a lightweight multi-scale model for high-accuracy pavement crack detection
[Paper](https://link.springer.com/article/10.1007/s11760-025-04064-0)
- Deep Learning-Based Lightweight Road Disease Detection Approach
[Paper](https://ieeexplore.ieee.org/abstract/document/11047837?casa_token=d6KPYRzBrpIAAAAA:_DdlO2FO6xZw3tdC6VhBZifjMChn8x8dfUCaJqypvp6VmlPWN5oCvg0BJUcIoV6GYBdUeRFOR6k)
- Lightweight Deep Convolutional Neural Network for Pavement Crack Recognition with Explainability Analysis
[Paper](https://link.springer.com/chapter/10.1007/978-3-031-82377-0_1)
- RT-DETR-Pothole: Lightweight Real-Time Detection Transformers for Improved Road Pothole Detection
[Paper](https://iopscience.iop.org/article/10.1088/1742-6596/3022/1/012003/meta)
- Advanced lightweight deep learning vision framework for efficient pavement damage identification
[Paper](https://www.nature.com/articles/s41598-025-97132-x)
- Real-time pavement distress detection based on deep learning and visual sensors
[Paper](https://www.tandfonline.com/doi/abs/10.1080/14680629.2024.2426034?casa_token=sPi3nlDcPeUAAAAA:iAYrFYH9qISq6FAdQDQaeOKdfdqGLfGe_YXJjpGVjhySnUQchj9LT219Yeuwwo6Fqlj9gx8TRWWFFgM)
- A Real-Time Road Damage Detection System for IoT Edge Devices Using Lightweight Deep Learning Models
[Paper](https://ieeexplore.ieee.org/abstract/document/11111694?casa_token=5A4IiiP1E60AAAAA:HqKob9Psngotd2XXwGGPAHvzSFuYYURDOKEsduy8j9kks1bkAGKcQJcR-AqetWFO7Xfcevbb4K8)
- Robust and Real-time Road Crack Detection through Collaborative Dual-Branch Learning on Robotic Sensing Platform
[Paper](https://ieeexplore.ieee.org/abstract/document/11139795?casa_token=x-Nw_FpdNjIAAAAA:GCDLZA6TA1woAwXjfJkwdd0mDGw91m6vDvAYtgi-sox_qwHu8G_ayKPKQMZOIMgWbwdPcrTJuqE)
> 2024
- Deployment strategies for lightweight pavement defect detection using deep learning and inverse perspective mapping
[Paper](https://www.sciencedirect.com/science/article/pii/S0926580524004187)
- Pavement Distress Detection Using YOLO and Faster RCNN on Edge Devices
[Paper](https://doi.org/10.1007/978-981-99-9342-0_26)
- YOLO9tr: a lightweight model for pavement damage detection utilizing a generalized efficient layer aggregation network and attention mechanism
[Paper](https://link.springer.com/article/10.1007/s11554-024-01545-2)
- Lightweight convolutional neural network driven by small data for asphalt pavement crack segmentation
[Paper](https://doi.org/10.1016/j.autcon.2023.105214)
- Design of a Real-time Detection System for Potholes and Bumps Using Deep Learning
[Paper](https://doi.org/10.1109/REM63063.2024.10735479)
- A Lightweight Network with Dual Encoder and Cross Feature Fusion for Cement Pavement Crack Detection
[Paper](https://doi.org/10.32604/cmes.2024.048175)
- Research on Rapid Detection and Identification Technology of Pavement Defects based on Target Detection Algorithm
[Paper](https://ieeexplore.ieee.org/abstract/document/10958086?casa_token=hHBqMgDafkwAAAAA:Dbdu_y_HAYAKu6Wu-vD4tmrv3cvXGYc3WVManGiCBmS2lUwD8xRNz2DQ4wAUonA2FYM7Ql3Zr6w)
- Cutting-Edge Deep Learning Models for Real-Time Pothole Detection and Localization in Challenging Road Conditions
[Paper](https://ieeexplore.ieee.org/abstract/document/10581208?casa_token=2PcxI0zTrvoAAAAA:CxS43aIHqtTS2XIuK1IrfOwWA7hS_zyU1eDlyTpjauSYl6UBpNVlNKTuGIcSs2aIiq89ExsDSyc)
- Multi-Scale Semantic Map Distillation for Lightweight Pavement Crack Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10565299?casa_token=TAc-KLSReLYAAAAA:zw8qq2oekZDlqXyBGAEhiOKmKIo9idIsnp9g1PpybSPQrvQpsfAY-uLX4if7Tq4E_UGkH3FNDAc)
- Real-Time Road Analysis System using Mems Data with IoT Application
[Paper](https://ieeexplore.ieee.org/abstract/document/10493936?casa_token=xsTgbr7_vFsAAAAA:NC4BbLvZsCuViT8K5PDJaThKFm_A2EbhChKm9Mo9J9Ql8dvhL6h1Scmf8sbbmOUl-3MGy0rS2p8)
- A lightweight detection method of pavement potholes based on binocular stereo vision and deep learning
[Paper](https://doi.org/10.1016/j.conbuildmat.2024.136733)
- A lightweight ground crack rapid detection method based on semantic enhancement
[Paper](https://www.cell.com/heliyon/fulltext/S2405-8440(24)10813-4)
- A lightweight feature attention fusion network for pavement crack segmentation
[Paper](https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13225)
- AI-Driven Approach for Automated Real-Time Pothole Detection, Localization, and Area Estimation
[Paper](https://journals.sagepub.com/doi/full/10.1177/03611981241246993?casa_token=TMvgUIe9e7UAAAAA%3Ac92v84aSxZxQJJg67ve51UqfR75A8wuYPvVA0UPhUMN6dcSrCRW5Eml0T9xJz6ZrWOQZHYGBO9ChaPw)
- Real-Time Pavement Crack Detection Based on Artificial Intelligence
[Paper](https://d1wqtxts1xzle7.cloudfront.net/118702392/3776-libre.pdf?1728261002=&response-content-disposition=inline%3B+filename%3DReal_Time_Pavement_Crack_Detection_Based.pdf&Expires=1770452376&Signature=XfPmjK-9srKQxgOkPhb1J9kIwZ6QfSIlXEZWkpk0Jh9T58b~ytDpwTWoz8Y5Q96WNAzM5xCz~fmXcaBGrvcRidkf9cruInDyTMqbwn754O8OQe4nLJpWlb9Vvu-P8ipwrSqoaNV2jtA2E9vB6M1GpYlmGb4uT4-QfzM-XTh~y-4ka60g6F9HiC1wtQb2A0eKHz9Qhnv2K~9Fr656iqgjzlFeU42JFFqFMVor5QsGlEJO0sG0aF~OroPLNQ5UyRzg5TnPc1p-OdEY9LPjghzQIeBgTvoNJd4IsDtkOw0J4XyXvNdNwVRbQ6iP9JNoSyjyWO0M7AYPW5qFUDj-Nx-aCA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA)
- A Real-Time Road Surface Identification and Tracking Based on Raspberry PI Assisted IoT Enabled Image Processing Scheme
[Paper](https://ieeexplore.ieee.org/abstract/document/10568900)
- Real Time Road Lane Detection and Vehicle Detection on YOLOv8 with Interactive Deployment
[Paper](https://ieeexplore.ieee.org/abstract/document/10847549?casa_token=nq64qQn-JBoAAAAA:GYmWOYSp7niyDnwNin8LiadcT3peSOFKEdeRZano_HGO000nQyYQrG_TT2eYQlXc7fm4WX8GR7w)
> 2023
- Vision Image Monitoring on Transportation Infrastructures: A Lightweight Transfer Learning Approach
[Paper](https://ieeexplore.ieee.org/abstract/document/9716851?casa_token=1LOz5ht7akgAAAAA:qA0MYCfmVjTxzteJP5IY1Nojn7pkNuELtuR8scmNKSSfVbzt8d-SnzPtb_uEzP0-nPGWZAR8BPk)
- Real time road defect monitoring from UAV visual data sources
[Paper](https://doi.org/10.1145/3594806.3596561)
- Automated Detection of Pavement Manhole on Asphalt Pavements with an Improved YOLOX
[Paper](https://doi.org/10.1061/JITSE4.ISENG-2313)
- ECSNet: An Accelerated Real-Time Image Segmentation CNN Architecture for Pavement Crack Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10214472?casa_token=zh-kNhlW5skAAAAA:7v25X4C0ecZjW6DstxJDeNh-Ii8AvmH005MQMqCKRqmxKJP4P5TRzEsP2NhMJg48LcBzVikwnsc)
- Lightweight computer vision based asphalt pavement crack visualization detection
[Paper](https://ieeexplore.ieee.org/abstract/document/10248599?casa_token=W527RjSEuioAAAAA:vH0TC6DX1JsKhzMOy0CEPcPHy6_kq0lbV_wVB3oppcd0I5YKlYo4zCQiAurohcmr3Xp_RbB9XEI)
- Road Marking Real-Time Detection with a Single Stage Object Detector
[Paper](https://ieeexplore.ieee.org/abstract/document/10398733?casa_token=59n9XkjVk-sAAAAA:a-TsaNy0_FnNW3hicN4leE0cHuXt0d3KBlMRdxjys39Ab9IWYyW30sS-gbRrA-oBa3qZN7mdICM)
- A Lightweight Road Crack and Damage Detection Method Using Yolov5s for IoT Applications
[Paper](https://ieeexplore.ieee.org/abstract/document/10233422?casa_token=o75i5w6SGYMAAAAA:onZKfnnAmxWRjyyH-f2Sb8qzwucr1fOF_l20J7KtE3cBh9nBL7OhEn1NLRh2TnJCP_p-j6AWgYU)
> 2022
- Efficient pavement distress classification via deep patch soft selective learning and knowledge distillation
[Paper](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ell2.12570)
- LLDNet: A Lightweight Lane Detection Approach for Autonomous Cars Using Deep Learning
[Paper](https://doi.org/10.3390/s22155595)
- RIIAnet: A Real-Time Segmentation Network Integrated with Multi-Type Features of Different Depths for Pavement Cracks
[Paper](https://doi.org/10.3390/app12147066)
- Improvement of Lightweight Convolutional Neural Network Model Based on YOLO Algorithm and Its Research in Pavement Defect Detection
[Paper](https://doi.org/10.3390/s22093537)
- Pothole Detection Using Deep Learning: A Real-Time and AI-on-the-Edge Perspective
[Paper](https://doi.org/10.1155/2022/9221211)
- A Lightweight Selective Feature Fusion and Irregular-Aware Network for Crack Detection Based on Federated Learning
[Paper](https://ieeexplore.ieee.org/abstract/document/9991509)
- Implementation of a Real-time Uneven Pavement Detection System on FPGA Platforms
[Paper](https://ieeexplore.ieee.org/abstract/document/9869054?casa_token=9Vp5KSuFCPEAAAAA:ePhhAtVJTsjFu-1ucP6SrN3jXxagtfZJh5EcR8mLxr-2BhiRlm3WNcpW_hpT15zKy8cVY43nJGI)
- LightAUNet: A Lightweight Fusing Attention Based UNet for Crack Detection
[Paper](https://ieeexplore.ieee.org/abstract/document/9886163?casa_token=NC-NqvwoUbkAAAAA:FDa1yqIbjxHO96OOkd5j6wFyAtrz09f_rV_aFU5QyP3VgO3OCnNpyzbRDfMnMcx5fzuYjRwQEPE)
- Research on intelligent inspection technologies of expressway maintenance based on edge computing
[Paper](https://ieeexplore.ieee.org/abstract/document/9927651?casa_token=0ulKjBrw-LsAAAAA:_lV6ZTGDfN1u18Mbb76JSCvqxMRFQJdwisXRiMf7KNy5nvj23BBJAc5pONHzZBHoaFzNzsbWlVg)
- Real-Time Sidewalk Crack Identification and Classification based on Convolutional Neural Network using Thermal Images
[Paper](https://ieeexplore.ieee.org/abstract/document/10029202?casa_token=4XxkmaIKyDoAAAAA:zoDeMMEKjQyQ8JLi5CU0UlGx4OFbmYVoJxIluhHGJReKSpdihsvEBdGhz9OHk6R8FbGCO0KIQvs)
> 2021
- Deep Learning-Based Real-Time Crack Segmentation for Pavement Images
[Paper](https://doi.org/10.1007/s12205-021-0474-2)
- Automatic pixel-level pavement crack recognition using a deep feature aggregation segmentation network with a scse attention mechanism module
[Paper](https://doi.org/10.3390/s21092902)
- Implementation of pavement defect detection system on edge computing platform
[Paper](https://doi.org/10.3390/app11083725)
- MobileCrack: Object Classification in Asphalt Pavements Using an Adaptive Lightweight Deep Learning
[Paper](https://doi.org/10.1061/JPEODX.0000245)





