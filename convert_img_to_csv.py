from PIL import Image
import numpy as np
import os

# read all the image file (myDir directory just have image)
def createFileList(myDir):
    fileList = []
    print(myDir)
    for root, dirs, files in os.walk(myDir, topdown=False):
        for name in files:
            fullName = os.path.join(root, name)
            fileList.append(fullName)
    return fileList


# create and open file csv
f = open('Your-csv-directory-you-want-to-save', 'w')

# create 1st row
f.write('emotion,pixels\n')

for name in os.listdir('Your-image-dir'):

    my_dir = 'Your-image-dir\\' + name
    myFileList = createFileList(my_dir)

    s = 48

    for file in myFileList:
        print(file)
        img_file = Image.open(file)
        img_file = img_file.resize((s, s))

        # get original image size
        width, height = img_file.size

        # Make image Greyscale
        img_grey = img_file.convert('L')

        # Save Greyscale values
        values = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))

        # 2d arr to 1d arr
        values = values.flatten()

        label = name

        # format csv file
        f.write(str(label) + ',')
        push = ""
        for value in values:
            push += (str(value) + ' ')

        f.write(push[:-1])
        f.write('\n')