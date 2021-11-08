
from matplotlib import pyplot
from matplotlib.patches import Rectangle

import imageIO.png
import math
class Queue:
   def __init__(self):
       self.items=[]
  
   def isEmpty(self):
       return self.items==[]
      
   def enqueue(self, item):
       self.items.insert(0,item)
      
   def dequeue(self):
       return self.items.pop()
      
   def size(self):
       return len(self.items)

def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):

    new_array = [[initValue for x in range(image_width)] for y in range(image_height)]
    return new_array


# this function reads an RGB color png file and returns width, height, as well as pixel arrays for r,g,b
def readRGBImageToSeparatePixelArrays(input_filename):

    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# This method packs together three individual pixel arrays for r, g and b values into a single array that is fit for
# use in matplotlib's imshow method
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(len(greyscale_pixel_array)):
        for x in range(len(greyscale_pixel_array[i])):
            greyscale_pixel_array[i][x] = round(pixel_array_r[i][x] * 0.299 + pixel_array_g[i][x] * 0.587 + pixel_array_b[i][x] * 0.114)
    
    return greyscale_pixel_array

def computeVerticalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    f = [[0 for i in range(image_width)]for p in range(image_height)]
    for x in range(1, image_height-1):
        for y in range(1, image_width - 1):
            f[x][y] = (pixel_array[x-1][y - 1] *(-1)+ pixel_array[x][y-1] * (-2)+ pixel_array[x + 1][y - 1] * (-1) + pixel_array[x-1][y+1] + pixel_array[x][y+1] * 2 + pixel_array[x+1][y+1]) /8
    return f     
def computeHorizontalEdgesSobelAbsolute(pixel_array, image_width, image_height):
    f = [[0 for i in range(image_width)]for p in range(image_height)]
    for x in range(1, image_height-1):
        for y in range(1, image_width - 1):
            f[x][y] = (pixel_array[x-1][y - 1] + pixel_array[x-1][y] * (2)+ pixel_array[x -1][y + 1]  + pixel_array[x+1][y-1] *(-1)+ pixel_array[x+1][y] * (-2) + pixel_array[x+1][y+1] *(-1)) /8
    return f
def computeMinAndMaxValues(pixel_array, image_width, image_height):
    m = []
    n = []
    for i in pixel_array:
        m.append(max(i))
        n.append(min(i))
    f = (max(m), min(n))
    return f


def scaleTo0And255AndQuantize(pixel_array, image_width, image_height):
    x = createInitializedGreyscalePixelArray(image_width, image_height)
    y = computeMinAndMaxValues(pixel_array, image_width, image_height)
    for i in range(len(pixel_array)):
        for z in range(len(pixel_array[i])):
            if max(y) - min(y) == 0:
                return x
            else:
                x[i][z] = round((pixel_array[i][z] - min(y)) * (255 / (max(y) - min(y))))
                
    return x

def computeEdgeMagnitude(p_array, image_width, image_height, v_scale, h_scale):
    f = [[0 for i in range(image_width)]for p in range(image_height)]
    for x in range(image_height):
        for y in range(image_width ):
            f[x][y] = math.sqrt(v_scale[x][y]**2 + h_scale[x][y]**2)
    return f


def fsmooth(pixel_array, image_width, image_height):     
    z= [[0 for x in range(image_width)] for y in range(image_height)]   
    for i in range(1, image_height-1):         
        for a in range(1, image_width-1):             
            z[i][a] = (pixel_array[i + 1][a +1] + pixel_array[i - 1][a - 1] + pixel_array[i][a - 1] + pixel_array[i + 1][a - 1] + pixel_array[i - 1][a + 1] + pixel_array[i][a + 1]) / 9     
    z = scaleTo0And255AndQuantize(z, image_width, image_height)    
    return z

def computeThresholdGE(pixel_array, threshold_value, image_width, image_height):
    z = createInitializedGreyscalePixelArray(image_width, image_height)
    for i in range(image_height):
        for j in range(image_width):
            if pixel_array[i][j] < threshold_value:
                z[i][j] = 0
            else:
                z[i][j] = 255
    return z
def computeErosion8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    z=createInitializedGreyscalePixelArray(image_width, image_height)
    for x in range(1, image_height-1):
        for y in range(1,image_width-1):
            if pixel_array[x-1][y-1]==0 or pixel_array[x-1][y]==0 or pixel_array[x-1][y+1]==0 or pixel_array[x][y-1]==0 or pixel_array[x][y]==0 or pixel_array[x][y+1]==0 or pixel_array[x+1][y-1]==0 or pixel_array[x+1][y]==0 or pixel_array[x+1][y+1]==0:
                z[x][y]=0
            else:
                z[x][y]=1
    return z
def computeDilation8Nbh3x3FlatSE(pixel_array, image_width, image_height):
    z=createInitializedGreyscalePixelArray(image_width, image_height)
    li = [1,1,1,1,1,1,1,1,1]
    for x in range(image_height):
        for y in range(image_width):
            l = []
            for i in range(x-1,x + 2):
                for j in range(y - 1, y + 2):
                    if i < 0 or j < 0 or i > image_height - 1 or j > image_width - 1:
                        l.append(0)
                    else:
                        l.append(pixel_array[i][j])
            for ele in range(len(l)):
                if l[ele] == li[ele] or l[ele] != 0 :
                    z[x][y] = 1
                    break
    return z

def computeConnectedComponentLabeling(pixel_array, image_width, image_height):
    r = createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0)
    visited = createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0)
    q = Queue()
    dic = {}
    k = 1 #component index
    for x in range(image_height):
        for y in range(image_width):
            c = 0
            if pixel_array[x][y] == 1 and visited[x][y] != 1:
                q.enqueue((x, y))
            while q.isEmpty() == False:
                c += 1
                current = q.dequeue()
                x_val = current[0]
                y_val = current[1]
                r[x_val][y_val] = k
                visited[x_val][y_val] = 1

                #below pixel
                if x_val+1 < image_height and pixel_array[x_val+1][y_val] == 1 and visited[x_val+1][y_val] == 0:
                    q.enqueue((x_val+1,y_val))
                    visited[x_val+1][y_val] = 1
                #above pixel
                if x_val-1 > 0 and pixel_array[x_val-1][y_val] == 1 and visited[x_val-1][y_val] == 0:
                    q.enqueue((x_val-1,y_val))
                    visited[x_val-1][y_val] = 1
                #left pixel
                if y_val-1 > 0 and pixel_array[x_val][y_val-1] == 1 and visited[x_val][y_val-1] == 0:
                    q.enqueue((x_val,y_val-1))
                    visited[x_val][y_val-1] = 1

                #right pixel
                if y_val+1 < image_width and pixel_array[x_val][y_val+1] == 1 and visited[x_val][y_val+1] == 0:
                    q.enqueue((x_val,y_val+1))
                    visited[x_val][y_val+1] = 1
            dic[k] = c #number of pixels
            k += 1
            visited[x][y] = 1
    return r, dic

def final_component(p1,  image_width, image_height, key):
    z = createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0)
    for i in range(image_height):
        for j in range(image_width):
            if p1[i][j] == key:
                z[i][j] = 1
    return z
def calculateboxbound(fcomp, image_width, image_height):
    minx, miny, maxx, maxy = image_height, image_width, 0, 0
    for j in range(image_height):
        for i in range(image_width):
            if fcomp[j][i] == 1:
                if minx > i:
                    minx = i
                if miny > j:
                    miny = j
                if maxx < i:
                    maxx = i
                if maxy < j:
                    maxy = j
    return minx, maxx, miny, maxy
def prepareRGBImageForImshowFromIndividualArrays(r,g,b,w,h):
    rgbImage = []
    for y in range(h):
        row = []
        for x in range(w):
            triple = []
            triple.append(r[y][x])
            triple.append(g[y][x])
            triple.append(b[y][x])
            row.append(triple)
        rgbImage.append(row)
    return rgbImage
    

# This method takes a greyscale pixel array and writes it into a png file
def writeGreyscalePixelArraytoPNG(output_filename, pixel_array, image_width, image_height):
    # now write the pixel array as a greyscale png
    file = open(output_filename, 'wb')  # binary mode is important
    writer = imageIO.png.Writer(image_width, image_height, greyscale=True)
    writer.write(file, pixel_array)
    file.close()



def main():
    filename = "./images/covid19QRCode/challenging/bloomfield.png"

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(filename)
    
    
    #gray scale
    z = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b,image_width, image_height)

    #ver and hori 
    ver = computeVerticalEdgesSobelAbsolute(z, image_width, image_height)
    hor = computeHorizontalEdgesSobelAbsolute(z, image_width, image_height)
    

    #magnitude
    magnitude = computeEdgeMagnitude(z, image_width, image_height, ver, hor)

    #smooth
    smooth = fsmooth(magnitude, image_width, image_height)
    for i in range(8):
        smooth = fsmooth(smooth, image_width, image_height)
    
    #threshold
    threshold = computeThresholdGE(smooth, 70, image_width, image_height)

    #fillholes
    dil = computeDilation8Nbh3x3FlatSE(threshold, image_width, image_height)
    ero = computeErosion8Nbh3x3FlatSE(dil, image_width, image_height)

    #connect component
    p_array, a_dict =  computeConnectedComponentLabeling(ero, image_width, image_height)
    max_pixels = max(a_dict.values())
    keys = [x for x, y in a_dict.items() if y == max_pixels]
    f_key = keys[0]

    f_comp = final_component(p_array,  image_width, image_height, f_key)


    #green box
    xmin, xmax, ymin, ymax = calculateboxbound(f_comp, image_width, image_height)
    start_p = (xmin, ymin)
    width = ymax - ymin
    height = xmax - xmin
    
    
    pyplot.imshow(prepareRGBImageForImshowFromIndividualArrays(px_array_r,px_array_g,px_array_b,image_width,image_height))



    

    # get access to the current pyplot figure
    axes = pyplot.gca()
    # create a 70x50 rectangle that starts at location 10,30, with a line width of 3
    rect = Rectangle( start_p, width, height, linewidth=3, edgecolor='g', facecolor='none' )
    # paint the rectangle over the current plot
    axes.add_patch(rect)



    # plot the current figure
    pyplot.show()



if __name__ == "__main__":
    main()





