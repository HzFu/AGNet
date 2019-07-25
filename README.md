# Attention Guided Network for Retinal Image Segmentation (AG-Net)
The code of "Attention Guided Network for Retinal Image Segmentation" in MICCAI 2019.


  - The code is based on: Python 2.7 + pytorch 0.4.0.
  - You can run <AG_Net\_path>/code/test.py for testing any new image directly.
  - You can run <AG_Net\_path>/code/main.py for training a new model.

## Quick usage on your data:
  - Put your desired file in "\<AG\_Net\_path\>/data/\<your\_file\_name\>".
  - Put the images in "\<AG\_Net\_path\>/data/\<your\_file\_name\>/images".
  - Put the labels in "\<AG\_Net\_path\>/data/\<your\_file\_name\>/label". 
  - Divide data into training and test data, and store the image name in the "train\_dict.pkl" file. (We provide a 'train\_dict.pkl' sample for DRIVE dataset)
  - The "train\_dict.pkl" should contains two dictionary: 'train\_list' and 'test\_list'.

Train your model with:
```sh
python main.py --data_path '../data/your_file_name'
```


##Reference:
1. S. Zhang, H. Fu, Y. Yan, Y. Zhang, Q. Wu, M. Yang, M. Tan, Y. Xu, "Attention Guided Network for Retinal Image Segmentation," in MICCAI, 2019.
2. H. Fu, J. Cheng, Y. Xu, D. W. K. Wong, J. Liu, and X. Cao, “Joint Optic Disc and Cup Segmentation Based on Multi-Label Deep Network and Polar Transformation,” IEEE Trans. Med. Imaging, vol. 37, no. 7, pp. 1597–1605, 2018.
