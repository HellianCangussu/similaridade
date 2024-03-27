from io import BytesIO
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import cv2
import numpy as np
from skimage import metrics
from matplotlib import pyplot as plt
from pdf2image import convert_from_path, convert_from_bytes

app = FastAPI()

def carregar_imagem_para_array(file):
    npimg=np.frombuffer(file,np.uint8)
    
    frame=cv2.imdecode(npimg,cv2.IMREAD_COLOR)
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

def ssim(image1, image2):
    # Load images
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation = cv2.INTER_AREA)
    # print(image1.shape, image2.shape)
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # Calculate SSIM
    ssim_score = metrics.structural_similarity(image1_gray, image2_gray, full=True)
    print("SSIM:", round(ssim_score[0], 2))
    return round(ssim_score[0], 2)

def orb(image1, image2):
    orb = cv2.ORB_create()
    
    kp_a = orb.detect(image1,None)
    kp_a, des_a = orb.compute(image1, kp_a)
    img1 = cv2.drawKeypoints(image1, kp_a, None, color=(0,255,0), flags=0)
    # plt.imshow(img1), plt.show()  

    kp_a = orb.detect(image2,None)
    kp_a, des_b = orb.compute(image2, kp_a)
    img2 = cv2.drawKeypoints(image2, kp_a, None, color=(0,255,0), flags=0)
    # plt.imshow(img2), plt.show()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0

    print("ORB:", len(similar_regions) / len(matches))
    return len(similar_regions) / len(matches)


@app.post("/")
async def root(imagem1: UploadFile, imagem2: UploadFile):
    print(imagem1.content_type, imagem2.content_type)
    imagem1_carregada = carregar_imagem_para_array(await imagem1.read())
    imagem2_carregada = carregar_imagem_para_array(await imagem2.read())
    resultadoSsim = ssim(imagem1_carregada, imagem2_carregada)
    resultadoOrb = orb(image1=imagem1_carregada, image2=imagem2_carregada)
    return { "SSIM": resultadoSsim, "ORB": resultadoOrb }

@app.post("/converter-pdf")
async def converter_pdf(pdf: UploadFile, posicao: int):
    images = convert_from_bytes(await pdf.read())
    filtered_image = BytesIO()
    images[posicao].save(filtered_image, "JPEG")
    filtered_image.seek(0)
    return StreamingResponse(filtered_image, media_type="image/jpeg")
    
@app.post("/numero-imagens-por-pdf")
async def obter_quantidade_imagens(pdf: UploadFile):
    images = convert_from_bytes(await pdf.read())
    return len(images)
