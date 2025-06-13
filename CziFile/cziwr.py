import tifffile
import czifile
filepath = './*.czi'
image = imread(filepath)
print(image.shape)
