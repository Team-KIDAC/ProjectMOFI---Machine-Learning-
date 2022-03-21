from __future__ import absolute_import, division, print_function, unicode_literals

from ClassifierMOFI import ClassifierMOFI

print("\t*****  MOFI - CONSOLE APPLICATION VIEW ******")

reAsk = True
modelMOFI = ClassifierMOFI()
class_names = ['St_001', 'St_002', 'St_003', 'St_004', 'St_005']  # Array that stores names of the classes
# path_to_train = os.path.abspath('DataSet/Testing/20200970_43.jpg')
while reAsk:
    try:
        imageFilePath = input("\n\tEnter the path to the image file: ")
        label = modelMOFI.predict(imageFilePath)
        print("PREDICTED IMAGE STATUS : ", class_names[label])
        print("MODEL CONFIDENCE : ", modelMOFI.probabilty, " %")

        information = [class_names[label], modelMOFI.probabilty]
        with open('MOFI_prediction.txt', 'w') as f:
            for item in information:
                f.write("%s\n" % item)

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
