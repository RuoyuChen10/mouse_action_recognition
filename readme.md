# Mouse Action Recognition based on Deep Learning and Video

![mouse action](https://github.com/RuoyuChen10/mouse_action_recognition/blob/master/img/mouse_action.gif)



Mouse action is important for drug analysis, which contribute to scientists to use the daily action of rats to conduct big data analysis. However, this is difficult for feature engineering due to complex scenes, so I want to through this project to understand the deep learning using for video analytics.

Requiring Environment:

- Pytorch 1.6.0
- numpy 1.18.1
- opencv-python 4.4.0.44

## Datasets path generate

First need data load, however, there is all kinds of datasets,different datasets may have different type to express. In this section, we try to write a txt to record the dir. For each line will have different action image serial, maybe consist of N images. We can write these images to one line, each image dir split by ' ', the end is the class of the action, using number to express. Each line will end up with '/n', don't forget removed when read the txt. If there is 16 images in one action it may express as follow:

```bash
./datasets/8/2/240.jpg ./datasets/8/2/241.jpg ./datasets/8/2/242.jpg ./datasets/8/2/243.jpg ./datasets/8/2/244.jpg ./datasets/8/2/245.jpg ./datasets/8/2/246.jpg ./datasets/8/2/247.jpg ./datasets/8/2/248.jpg ./datasets/8/2/249.jpg ./datasets/8/2/250.jpg ./datasets/8/2/251.jpg ./datasets/8/2/252.jpg ./datasets/8/2/253.jpg ./datasets/8/2/254.jpg ./datasets/8/2/255.jpg 8
./datasets/7/7/27806.jpg ./datasets/7/7/27807.jpg ./datasets/7/7/27808.jpg ./datasets/7/7/27809.jpg ./datasets/7/7/27810.jpg ./datasets/7/7/27811.jpg ./datasets/7/7/27812.jpg ./datasets/7/7/27813.jpg ./datasets/7/7/27814.jpg ./datasets/7/7/27815.jpg ./datasets/7/7/27816.jpg ./datasets/7/7/27817.jpg ./datasets/7/7/27818.jpg ./datasets/7/7/27819.jpg ./datasets/7/7/27820.jpg ./datasets/7/7/27821.jpg 7
./datasets/0/5/3702.jpg ./datasets/0/5/3703.jpg ./datasets/0/5/3704.jpg ./datasets/0/5/3705.jpg ./datasets/0/5/3706.jpg ./datasets/0/5/3707.jpg ./datasets/0/5/3708.jpg ./datasets/0/5/3709.jpg ./datasets/0/5/3710.jpg ./datasets/0/5/3711.jpg ./datasets/0/5/3712.jpg ./datasets/0/5/3713.jpg ./datasets/0/5/3714.jpg ./datasets/0/5/3715.jpg ./datasets/0/5/3716.jpg ./datasets/0/5/3717.jpg 0
./datasets/0/1/347.jpg ./datasets/0/1/348.jpg ./datasets/0/1/349.jpg ./datasets/0/1/350.jpg ./datasets/0/1/351.jpg ./datasets/0/1/352.jpg ./datasets/0/1/353.jpg ./datasets/0/1/354.jpg ./datasets/0/1/355.jpg ./datasets/0/1/356.jpg ./datasets/0/1/357.jpg ./datasets/0/1/358.jpg ./datasets/0/1/359.jpg ./datasets/0/1/360.jpg ./datasets/0/1/361.jpg ./datasets/0/1/362.jpg 0
```

There will generate 3 txt: `train.txt`, `val.txt` and `test.txt`, using for training set, validation set and test set. The details can visit the 3 txt.

Code `generate_txt.py` showed one example for generate the txt, however there are changes according to the data set format. We can according to our datasets to generate the same format that can adopt to this program.

## Training the network

The network's input is 112\*112, if is not will be auto resize to 112\*112, the class num changed `utils.py` parameter `class_num`, this will set 16.

the details training parameters in `train.py`, it can change when tuning, if need training networks, try:

```bash
python train.py
```

Then it will begin to train if you see the list as:

```bash
Original data load finished!
loss=2.772338
loss=2.771795
loss=2.772871
loss=2.772724
Test set: Average loss: 1.38636017, Accuracy: 0/2 (0%)
loss=2.772496
loss=2.772238
loss=2.772055
loss=2.771399
Test set: Average loss: 1.38633895, Accuracy: 0/2 (0%)
loss=2.772418
loss=2.771773
loss=2.772142
loss=2.771398
Test set: Average loss: 1.38631725, Accuracy: 0/2 (0%)
loss=2.772377
loss=2.771513
loss=2.771619
loss=2.772004
```

