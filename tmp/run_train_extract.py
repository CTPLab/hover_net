import cv2
import matplotlib.pyplot as plt
from pathlib import Path


bern_folder = Path('/home/histopath/Data/hover')
tma_folder = bern_folder / 'pred'
for tma_id, tma_file in enumerate(list(tma_folder.glob('**/*.png'))):
    print(tma_file)
    tma = cv2.imread(str(tma_file))
    tma_gray = cv2.cvtColor(tma, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        tma_gray, 200, 255, cv2.THRESH_BINARY_INV)[1]

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    tma_roi = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        tma_roi = tma[y:y+h, x:x+w]
        break

    if (tma_roi is not None) and tma_roi.shape[0] >= 1024 and tma_roi.shape[1] >= 1024:
        tma_h, tma_w = tma_roi.shape[:2]
        tma_small = tma_roi[tma_h // 2 - 256:tma_h // 2 + 256, tma_w // 2 - 256: tma_w // 2 + 256, :]
        print(tma_file.stem, tma_small.shape)
        tma_name = bern_folder / 'output' / '{}_roi.png'.format(tma_file.stem)
        cv2.imwrite(str(tma_name), tma_small)
        print('generate {} with shape {} done!'.format(
            tma_file.stem + '_roi.png', tma_roi.shape))

    # if tma_id == 1:
    #     break
