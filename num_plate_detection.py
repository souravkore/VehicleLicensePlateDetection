import cv2
import numpy as np
from PIL import Image
import pytesseract as tess


tess.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


def preproces(resize_img):

    cv2.imshow("resized original image", resize_img)
    # applying the blur to remove the noise from the image
    img_blur = cv2.blur(resize_img, (4, 4))
    # cv2.waitKey(0)
    cv2.imshow("img blur", img_blur)

    # changing the color from RGB to grayscale to reduce extra computation
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)
    # cv2.waitKey(0)
    cv2.imshow("img grayscale", img_gray)

    return img_gray


def thresholding(gray_img):

    ret, img_threshold = cv2.threshold(gray_img, 127, 255, 0)
    # cv2.waitKey(0)
    cv2.imshow("threshold img", img_threshold)

    return img_threshold


def cannyedge(thresh_img):
    canny_edge_img = cv2.Canny(thresh_img, 170, 200)
    # cv2.waitKey(0)
    cv2.imshow("canny edge img", canny_edge_img)
    return canny_edge_img


def contouring(canny_edge):
    img_copy1 = img_resize.copy()

    contours, hierarchy = cv2.findContours(canny_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:]

    cv2.drawContours(img_copy1, contours, -1, (0, 255, 0), 2)
    # cv2.waitKey(0)
    cv2.imshow("contour image ", img_copy1)

    print("number of contours present :- "+str(len(contours)))
    return contours


def detect_plate(contours_img):
    contours_img = contours_img[:]
    count = 0
    numberplate_contour = None
    for contour in contours_img:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*perimeter, True)

        # print("perimeter "+str(perimeter))
        # print("approx "+str(approx))

        if len(approx) == 4:
            # select the contour with 4 corners
            numberplate_contour = approx
            x, y, w, h = cv2.boundingRect(contour)
            new_cropped_image = img_resize.copy()[y:y + h, x:x + w]

            img2 = cv2.rectangle(img_resize.copy(), (x, y), (x+w, y+h), (0, 255, 0), 3)
            # cv2.waitKey(0)
            cv2.imshow("number_plate", img2)

            resize_new_cropped_image = cv2.resize(new_cropped_image, (420, 100))
            # cv2.waitKey(0)
            cv2.imshow(" detected number plate ", resize_new_cropped_image)

            # denoising_cropped_img = cv2.fastNlMeansDenoisingColored(resize_new_cropped_image, None, 10, 10, 7, 21)
            # cv2.imshow("denoising - ", denoising_cropped_img)

            # cv2.imwrite("cropped_Image_TEXT-"+str(count)+".png", new_cropped_image)

            count += 1
            break

    return resize_new_cropped_image
    # return denoising_cropped_img


def recog_plate(denoised_num_plate):

    img_number = Image.fromarray(denoised_num_plate)
    text = tess.image_to_string(img_number, lang='eng')

    print("\n\tDetected number plate :- "+text)


if __name__ == '__main__':
    print("DETECTING PLATE . . .")

    # loading/reading the image
    img = cv2.imread("car.jpg")

    # resizing the original image
    img_resize = cv2.resize(img, (620, 620))

    preprocessd_img = preproces(img_resize)

    threshold_img = thresholding(preprocessd_img)
    canny_edge = cannyedge(threshold_img)
    contours_img = contouring(canny_edge)
    num_plate = detect_plate(contours_img)
    recog_plate(num_plate)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

