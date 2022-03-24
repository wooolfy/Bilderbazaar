import torch
import cv2
import numpy
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
#from pymatting import cutout
from os import listdir
from os.path import isfile, join
import torchvision.models as models


mypath="./testpdf"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
#print(onlyfiles)



# Definition of the used Model. Resnet 101 seems to work best

model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)


#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
#model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

for i in range (len(onlyfiles)):
    try:
        # Get the image from the file system and load it into memory;
        input_image = Image.open('./testpdf/'+onlyfiles[i])

        #transfer imate to numpy-array
        np_image = numpy.array(input_image)

        #write back to temp file to mess with; png seems to work better than jpg
        #cv2.imwrite('document.png', np_image)
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            #a little less std values seem to get even better results. maybe below 2.5? mean=[0.485, 0.456, 0.506] std=[0.359, 0.354, 0.355]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # get the tensor ready to process the input image
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            try:
                input_batch = input_batch.to('cuda')
                model.to('cuda')
                print('Processing ', onlyfiles[i], 'on GPU')
            except Exception:
                print('Processing ', onlyfiles[i], 'on CPU')

        #produce output of model
        with torch.no_grad():
            output = model(input_batch)['out'][0]
        model_output = output.argmax(0)


        # create a color pallette, selecting a color for each class
        palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        #colors = torch.as_tensor([i for i in range(20)])[:, None] * palette
        #colors = (colors % 255).numpy().astype("uint8")

        # plot the semantic segmentation predictions of 21 classes in each color
        identifiedObjects = Image.fromarray(model_output.byte().cpu().numpy()).resize(input_image.size)
        #r.putpalette(colors)
        palettedata = [1, 1, 1, 2, 2, 2, 3, 3, 3, 255, 255, 255]
        identifiedObjects.putpalette(palettedata * 64)

        #transfer back to numpy array
        np_image = numpy.array(identifiedObjects)
        #todo: get size from original image. resolution is only temp
        resized = cv2.resize(np_image, (1717, 2317), interpolation=cv2.INTER_AREA)

        #write back to file
        cv2.imwrite('./outputmasks/mask_'+onlyfiles[i], resized)
        #cache leeren. sonst überlauf ... oder nur alle paar schleifenaufrufe weil schneller?
        torch.cuda.empty_cache()
    except Exception:
        print ('Error in run ',i+1,' : gpu ram ende')
        continue
cv2.imshow("Detected Edges", np_image)
cv2.waitKey(0)