from SideWindowFilter import process as swf
import numpy as np

if __name__ == '__main__':
    img_path = './img/ori.jpg'  # img path

    filter = np.array([[[[0.0453542, 0.0566406, 0.0453542],
                         [0.0566406, 0.0707355, 0.0566406],
                         [0.0453542, 0.0566406, 0.0453542]]]], dtype=np.float)
    filter /= np.sum(filter)  # Gauss filter

    # filter = np.array([[[[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]]], dtype=np.float)  # mean filter
    img = swf(img_path, filter=filter, iteration=1)
    # img.save('img/process.jpg')
