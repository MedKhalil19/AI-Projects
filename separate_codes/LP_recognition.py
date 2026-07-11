from tensorflow.keras.models import load_model
from numpy import array, ones, zeros, arange, uint8
from cv2 import reduce,copyMakeBorder,BORDER_CONSTANT,CV_32S,REDUCE_SUM,COLOR_BGR2GRAY,cvtColor,rectangle,resize,addWeighted,putText,FONT_HERSHEY_DUPLEX,FONT_HERSHEY_SIMPLEX
import numpy as np
import cv2
def histogram_of_pixel_projection(img):
    """
    This method is responsible for licence plate segmentation with histogram of pixel projection approach
    :param img: input image
    :return: list of image, each one contain a digit
    """
    # list that will contains all digits
    caracrter_list_image = list()

    # img = crop(img)

    # Add black border to the image
    BLACK = [0, 0, 0]
    img = copyMakeBorder(img, 3, 3, 3, 3, BORDER_CONSTANT, value=BLACK)

    # change to gray
    gray = cvtColor(img, COLOR_BGR2GRAY)

    # Change to numpy array format
    nb = array(gray)

    # Binarization
    nb[nb > 120] = 255
    nb[nb < 120] = 0

    # compute the sommation
    x_sum = reduce(nb, 0, REDUCE_SUM, dtype=CV_32S)
    y_sum = reduce(nb, 1, REDUCE_SUM, dtype=CV_32S)

    # rotate the vector x_sum
    x_sum = x_sum.transpose()

    # get height and weight
    x = gray.shape[1]
    y = gray.shape[0]

    # division the result by height and weight
    x_sum = x_sum / y
    y_sum = y_sum / x

    # x_arr and y_arr are two vector weight and height to plot histogram projection properly
    x_arr = arange(x)
    y_arr = arange(y)

    # convert x_sum to numpy array
    z = array(x_sum)

    # convert y_arr to numpy array
    w = array(y_sum)

    # convert to zero small details
    z[z < 15] = 0
    z[z > 15] = 1

    # convert to zero small details and 1 for needed details
    w[w < 20] = 0
    w[w > 20] = 1

    # vertical segmentation
    test = z.transpose() * nb

    # horizontal segmentation
    test = w * test

    # plot histogram projection result using pyplot
    #horizontal = plt.plot(w, y_arr)
    #plt.show()
    #vertical = plt.plot(x_arr ,z)
    #plt.show()
    #plt.show(horizontal)
    #plt.show(vertical)

    f = 0
    ff = z[0]
    t1 = list()
    t2 = list()
    for i in range(z.size):
        if z[i] != ff:
            f += 1
            ff = z[i]
            t1.append(i)
    rect_h = array(t1)

    f = 0
    ff = w[0]
    for i in range(w.size):
        if w[i] != ff:
            f += 1
            ff = w[i]
            t2.append(i)
    rect_v = array(t2)

    # take the appropriate height
    rectv = []
    rectv.append(rect_v[0])
    rectv.append(rect_v[1])
    max = int(rect_v[1]) - int(rect_v[0])
    for i in range(len(rect_v) - 1):
        diff2 = int(rect_v[i + 1]) - int(rect_v[i])

        if diff2 > max:
            rectv[0] = rect_v[i]
            rectv[1] = rect_v[i + 1]
            max = diff2

    # extract caracter
    for i in range(len(rect_h) - 1):

        # eliminate slice that can't be a digit, a digit must have width bigger then 8
        diff1 = int(rect_h[i + 1]) - int(rect_h[i])

        if (diff1 > 5) and (z[rect_h[i]] == 1):
            # cutting nb (image) and adding each slice to the list caracrter_list_image
            caracrter_list_image.append(nb[int(rectv[0]):int(rectv[1]), rect_h[i]:rect_h[i + 1]])

            # draw rectangle on digits
            rectangle(img, (rect_h[i], rectv[0]), (rect_h[i + 1], rectv[1]), (0, 255, 0), 1)

    # Show segmentation result
    #image = plt.imshow(img)
    #plt.show() ################################################################
    #plt.show(image)

    return caracrter_list_image
def fix_dimension(img):
  new_img = np.zeros((28,28,3))
  for i in range(3):
    new_img[:,:,i] = img
  return new_img
model = load_model(r'C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_Plate_Recognition\ocrmodel.h5')


def show_results(char):
    dic = {}
    characters = '0123456789T'
    for i, c in enumerate(characters):
        dic[i] = c

    output = []
    for i, ch in enumerate(char):  # iterating over the characters
        img_ = cv2.resize(ch, (28, 28))
        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3)  # preparing image for the model
        y_ = np.argmax(model.predict(img), axis=-1)[0]  # predicting the class
        # print(y_)
        character = dic[y_]  #
        if (character == "T"):
            output.append("Tunisie")
        else:
            output.append(character)  # storing the result in a list

    plate_number = ''.join(output)

    return output
def draw_text_on_image(img, title, text, x=150, y=250, w=500, h=100):
    # First, we crop the sub-rect from the image
    sub_img = img[y:y+h, x:x+w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.5, 0)

    # Putting the image back to its position
    img[y:y+h, x:x+w] = res
    cv2.putText(img, title, (x + 10, y + 30), cv2.FONT_HERSHEY_DUPLEX, 1, (128, 190, 82), 2)
    cv2.putText(img, text, (x + 10, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (118, 82, 26), 2)
    return img


def LP_recognition(img, new_img, top):
    """
    Recognizes and annotates the license plate from the provided image.

    :param img: Cropped image of the license plate.
    :param new_img: Image with a detection box indicating the license plate location.
    :param top: Vertical position for placing the recognition result on the image.
    :return: Annotated image with the license plate recognition result.
    """

    # Perform character segmentation
    characters = histogram_of_pixel_projection(img)

    # Check if characters were successfully segmented
    if not characters:
        print("No characters were segmented.")
        return new_img

    # Perform character recognition
    results = show_results(characters)

    # Concatenate results into a single plate number
    plate_text = ''.join(results)
    title = "Licence Plate:"

    # Define text box coordinates
    new_img_width = new_img.shape[1]
    new_img_height = new_img.shape[0]
    x = new_img_width // 2 - 200
    y = top
    w = new_img_width // 2 + 25
    h = 100

    # Annotate the image with recognition results
    final_img = draw_text_on_image(new_img, title, plate_text, x, y, w, h)

    return final_img


# Example usage
img = cv2.imread(r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Licence_Plate_extracted.jpg")
new_img = cv2.imread(r"C:\Users\khkha\OneDrive\Bureau\Vehicle-Recognition-System\Deep-Learning\Processed_Image.jpg")
top = 375  # Example value for top

result_img = LP_recognition(img, new_img, top)
cv2.imshow("Result", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
