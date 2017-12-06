from PIimport Image

im = Image.open('test.jpg')

height = 100
width = 100

im = im.resize((width, height), Image.ANTIALIAS)    # best down-sizing filter

ext = ".png"
im.save("ANTIALIAS" + ext)