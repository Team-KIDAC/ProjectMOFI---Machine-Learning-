from __future__ import absolute_import, division, print_function, unicode_literals

import traceback

from ClassifierMOFI import ClassifierMOFI

import urllib.request


def imagePath(image_url):
    modelMOFI = ClassifierMOFI()
    class_names = ['St_001', 'St_002', 'St_003', 'St_004', 'St_005']  # Array that stores names of the classes
    try:
        imageFilePath = urllib.request.urlopen(image_url)
        label = modelMOFI.predict(imageFilePath)
        # print("PREDICTED IMAGE STATUS : ", class_names[label])
        # print("MODEL CONFIDENCE : ", modelMOFI.probabilty, " %")
        if float(modelMOFI.probabilty) > 70:
            print("PREDICTED IMAGE STATUS : ", class_names[label])
        else:
            print("Unknown")
        # information = [class_names[label]]
        # with open('MOFI_prediction.txt', 'w') as f:
        #     for item in information:
        #         f.write("%s\n" % item)
        # printGraph = input("Do you want a graphical interface of the image? (Y/N): ")
        # if printGraph.lower() == "y":
        #     modelMOFI.printBarGraph()
        # reply = input("Try another face image? (Y/N): ")
        # reply = reply.lower()
        # if reply == "y":
        #     reAsk = True
        # else:
        #     print("Thank you for using the MOFI application.")
        #     reAsk = False
        #     break
    except:
        traceback.print_exc()
        print("Invalid input please try again...")


if __name__ == '__main__':
    imagePath('https://mofiblob.blob.core.windows.net/mofiimages/20200940_7.jpg')
