from __future__ import absolute_import, division, print_function, unicode_literals
from ClassifierMOFI import ClassifierMOFI

print("\t*****  MOFI - CONSOLE APPLICATION VIEW ******")

reAsk = True
modelMOFI = ClassifierMOFI()
class_names = ['St_001', 'St_002', 'St_003', 'St_004', 'St_005']  # Array that stores names of the classes
while reAsk:
    try:
        imageFilePath = input("Enter face image file path: ")
        label = modelMOFI.predict(imageFilePath)
        print("PREDICTED IMAGE STATUS : ", class_names[label])
        print("MODEL CONFIDENCE : ", modelMOFI.probabilty, " %")
        printGraph = input("Do you want a graphical interface of the image? (Y/N): ")
        if printGraph.lower() == "y":
            modelMOFI.printBarGraph()
        reply = input("Try another face image? (Y/N): ")
        reply = reply.lower()
        if reply == "y":
            reAsk = True
        else:
            print("Thank you for using the MOFI application.")
            reAsk = False
            break
    except:
        print("Invalid input please try again...")
