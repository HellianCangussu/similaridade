from io import BytesIO
import PIL
from fastapi import FastAPI
from fastapi import UploadFile
from fastapi.responses import FileResponse, StreamingResponse
import cv2
import numpy as np
from skimage import metrics
from pdf2image import convert_from_path, convert_from_bytes
import pytesseract
from PIL import Image

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
    # img1 = cv2.drawKeypoints(image1, kp_a, None, color=(0,255,0), flags=0)
    # plt.imshow(img1), plt.show()  

    kp_b = orb.detect(image2,None)
    kp_b, des_b = orb.compute(image2, kp_b)
    # img2 = cv2.drawKeypoints(image2, kp_b, None, color=(0,255,0), flags=0)
    # plt.imshow(img2), plt.show()

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_a, des_b)
    similar_regions = [i for i in matches if i.distance < 50]
    if len(matches) == 0:
        return 0

    print("ORB:", len(similar_regions) / len(matches))
    return len(similar_regions) / len(matches)

async def obter_texto_imagem(imagem):
    custom_config = r'--oem 3 --psm 6'
    return pytesseract.image_to_string(imagem, config=custom_config)
    
@app.post("/")
async def root(item1: UploadFile, item2: UploadFile):

    lista_comparacoes = []
    pdf_convertido_1 = []
    pdf_convertido_2 = []

    if (item1.content_type == "application/pdf"):
        pdf_convertido_1 = convert_from_bytes(await item1.read())
    else:
        pdf_convertido_1 = [Image.frombytes(item1.read())]

    if (item2.content_type == "application/pdf"):
        pdf_convertido_2 = convert_from_bytes(await item2.read())
    else:
        pdf_convertido_2 = [Image.frombytes(item2.read())]

    comparacoesIdxs = []

    for i, pdf1Image in enumerate(pdf_convertido_1):
        
        pdf_convertido_1_bytes = BytesIO()
        pdf1Image.save(pdf_convertido_1_bytes, "JPEG")

        for j, pdf2Image in enumerate(pdf_convertido_2):

            if (item1.filename == item2.filename and i == j):
                continue

            if ([i,j] in comparacoesIdxs or [j, i] in comparacoesIdxs):
                continue

            pdf_convertido_2_bytes = BytesIO()
            pdf2Image.save(pdf_convertido_2_bytes, "JPEG")
            pdf_convertido_1_bytes.seek(0)
            pdf_convertido_2_bytes.seek(0)
            imagem1_carregada = carregar_imagem_para_array(pdf_convertido_1_bytes.read())
            imagem2_carregada = carregar_imagem_para_array(pdf_convertido_2_bytes.read())

            textoImg1 = await obter_texto_imagem(imagem1_carregada)
            textoImg2 = await obter_texto_imagem(imagem1_carregada)

            if ('nota' not in textoImg1 and 'comprovante' not in textoImg1):
                continue

            if ('nota' not in textoImg1 and 'comprovante' not in textoImg2):
                continue

            resultadoSsim = ssim(imagem1_carregada, imagem2_carregada)
            resultadoOrb = orb(image1=imagem1_carregada, image2=imagem2_carregada)
            comparacoesIdxs.append([i,j])
            lista_comparacoes.append({ "SSIM": resultadoSsim, "ORB": resultadoOrb, "DocumentoBasePagina": i+1, "DocumentoComparadoPagina": j+1 })

    return lista_comparacoes 


