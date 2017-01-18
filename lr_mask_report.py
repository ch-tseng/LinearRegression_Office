#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#載入必要模組
from __future__ import print_function
from sklearn.metrics import classification_report
#from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2
import numpy as np

#使用參數方式傳入Training和Test的dataset
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--training", required=True, help="Path to the logos training dataset")
ap.add_argument("-t", "--test", required=True, help="Path to the test dataset")
args = vars(ap.parse_args())

#產生mask圖層
mask = np.zeros((500,500), dtype="uint8")
#定義四個角落的位置
cv2.rectangle(mask, (1, 63), (174, 262), 255, -1)
cv2.rectangle(mask, (1, 292), (180, 500), 255, -1)
cv2.rectangle(mask, (300, 60), (480, 265), 255, -1)
cv2.rectangle(mask, (300, 295), (480, 500), 255, -1)


#data用來存放HOG資訊，labels則是存放對應的標籤
data = []
labels = []

#依序讀取training dataset中的圖檔
for imagePath in paths.list_images(args["training"]):
        #將資料夾的名稱取出作為該圖檔的標籤
        make = imagePath.split("/")[-2]

        #----以下為訓練圖檔的預處理----# 
        #載入圖檔,轉為灰階,作模糊化處理
        #將Mask疊加到圖檔
        image = cv2.imread(imagePath)
        masked = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        #將裁切後的圖檔尺寸更改為500x500。
        Cutted = cv2.resize(gray, (500, 500))
        cv2.imshow("Cutted", Cutted)
        #----訓練圖檔預處理結束----#
        #取得其HOG資訊及視覺化圖檔
        (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        #將HOG資訊及標籤分別放入data及labels陣列
        data.append(H)
        labels.append(make)

#開始用KNN模型來訓練
#model = SVC(kernel="linear")
#model = SVC(kernel="poly", degree=2, coef0=1)
model = LogisticRegression()
#傳入data及labels陣列開始訓練
model.fit(data, labels)

testdata = []
testlabels = []
#準備使用Test Dataset來驗証
for (i, imagePath) in enumerate(paths.list_images(args["test"])):
        make = imagePath.split("/")[-2]
        testlabels.append(make)
        #從測試資料中讀取圖檔
        image = cv2.imread(imagePath)
        #----以下為測試圖檔的預處理----# 
        #轉為灰階並模糊化
        masked = cv2.bitwise_and(image, image, mask=mask)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        #將裁切後的圖檔尺寸更改為500x500。
        Cutted = cv2.resize(gray, (500, 500))
        cv2.imshow("Cutted", Cutted)
        #----圖檔預處理結束----#

        #取得其HOG資訊及視覺化圖檔
        (H, hogImage) = feature.hog(Cutted, orientations=9, pixels_per_cell=(10, 10),
                cells_per_block=(2, 2), transform_sqrt=True, visualise=True)

        testdata.append(H)

print(classification_report(testlabels, model.predict(testdata)))
