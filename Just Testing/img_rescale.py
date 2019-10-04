from PIL import Image
im = Image.open('black_star.jpg')

new_size = (200, 200)

# Image.resize(size, resample)
im1 = im.resize(new_size, resample=Image.NEAREST)
im2 = im.resize(new_size, resample=Image.BILINEAR)

print('Original size: ', im.size)
print('New size: ', im1.size)


im1.show()
im2.show()

# this just saves a copy of the scaled image to the directory, can change directory, testing
im1.save(r'C:\Users\selen\Pictures\NN_output.png', 'png')
im2.save(r'C:\Users\selen\Pictures\BL_output.png', 'png')
