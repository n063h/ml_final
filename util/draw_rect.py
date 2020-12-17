import cv2 as cv

def draw_anchor(img_path, box,label="",save_path=None):

    img = cv.imread(img_path)
    # real_size=[img.shape[1],img.shape[0]]
    # cv.rectangle(img, (box[0], box[2]), (box[1], box[3]), (255, 255, 255), thickness=2)
    # cv.putText(img, label, (box[0], box[2]), cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0),
    #            thickness=2)
    cv.imshow('head', img)
    if save_path!=None:
        cv.imwrite(save_path, img)  # save picture

if __name__ == '__main__':
    draw_anchor('./dataset/fabric_data_new\\trgt\\1594445240123_dev001\\80854798856_11_1_2.jpg',(125,188,171,230),"4")