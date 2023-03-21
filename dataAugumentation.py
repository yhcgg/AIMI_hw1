import os
from PIL import Image
import random

random.seed(10)

def makeDir(dirPath):
	if not os.path.exists(dirPath):
		os.mkdir(dirPath)


def generateDataset(datasetPath):
	augDirPath = datasetPath + '/augmentation'
	makeDir(augDirPath)

	for class_name in ['NORMAL', 'PNEUMONIA']:
		makeDir(augDirPath + '/' + class_name)

		img_list = os.listdir(datasetPath + '/train/' + class_name)
		random.shuffle(img_list)

		count = (len(img_list) * 2) // 3  # rotate:zoom = 2 : 1

		# rotate
		for imgName in img_list[:count]:
			img = Image.open(datasetPath + '/train/' + class_name + '/' + imgName)
			img = img.rotate(random.randint(-30, 30))
			img.save(augDirPath + '/' + class_name + '/' + imgName.split('.')[0] + '_aug.jpeg')

		# zoom
		for imgName in img_list[count:]:
			img = Image.open(datasetPath + '/train/' + class_name + '/' + imgName)
			scale = random.uniform(1, 2)
			ori_width, ori_height = img.width, img.height
			img = img.resize((int(ori_width * scale), int(ori_height * scale)))
			img = img.crop(((img.width - ori_width) // 2, (img.height - ori_height) // 2,
			                ((img.width - ori_width) // 2) + ori_width, ((img.height - ori_height) // 2) + ori_height))
			img.save(augDirPath + '/' + class_name + '/' + imgName.split('.')[0] + '_aug.jpeg')
