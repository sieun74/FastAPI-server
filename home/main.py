from fastapi import FastAPI, UploadFile, File,WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import io
import httpx
import traceback

app = FastAPI()

# CORS 허용 설정 (필요에 따라 도메인 변경 가능)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 예: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLIP 모델 및 프로세서 로드 (최초 1회만 인터넷 연결 필요)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

import traceback

@app.post("/image/walk")
async def check_walk_mission(file: UploadFile = File(...)):
    try:
        candidate_texts = ["사람이", "무관한 사진"]
        image_bytes = await file.read()
        print(f"[walk] 이미지 바이트 크기: {len(image_bytes)}")
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("[walk] 이미지 로드 성공")

        inputs = processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        top_idx = torch.argmax(probs).item()
        top_text = candidate_texts[top_idx]
        passed = "성공" if top_text == "산책 사진" else "실패"
        print(f"[walk] 판단 결과: {passed}")

        return JSONResponse(content={"mission_passed": passed})

    except Exception as e:
        print("[walk] 에러 발생:", e)
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/image/nature")
async def check_nature_mission(file: UploadFile = File(...)):
    try:
        candidate_texts = ["자연물 사진", "무관한 사진"]
        image_bytes = await file.read()
        print(f"[nature] 이미지 바이트 크기: {len(image_bytes)}")
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        print("[nature] 이미지 로드 성공")

        inputs = processor(text=candidate_texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        top_idx = torch.argmax(probs).item()
        top_text = candidate_texts[top_idx]
        passed = "성공" if top_text == "자연물 사진" else "실패"
        print(f"[nature] 판단 결과: {passed}")

        return JSONResponse(content={"mission_passed": passed})

    except Exception as e:
        print("[nature] 에러 발생:", e)
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

GROQ_API_KEY = 'grop_your_API'
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama3-70b-8192"

# 최대 대화 저장 수
MAX_HISTORY = 10

@app.websocket("/ws")
async def chat_with_groq(websocket: WebSocket):
    await websocket.accept()

    conversation = [
        {"role": "system", "content": "당신은 한국어를 잘하는 친절한 AI 챗봇입니다. 질문에 정확하게 답해주세요."}
    ]

    try:
        async with httpx.AsyncClient() as client:
            while True:
                user_input = await websocket.receive_text()
                conversation.append({"role": "user", "content": user_input})

                # 최근 대화만 유지
                conversation = conversation[-MAX_HISTORY:]

                # Groq API 호출
                response = await client.post(
                    GROQ_API_URL,
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": GROQ_MODEL,
                        "messages": conversation
                    },
                    timeout=30.0
                )

                if response.status_code != 200:
                    await websocket.send_text(f"❌ Groq API 호출 실패: {response.text}")
                    continue

                result = response.json()
                reply = result["choices"][0]["message"]["content"]
                conversation.append({"role": "assistant", "content": reply})

                await websocket.send_text(reply)

    except WebSocketDisconnect:
        print("🔌 클라이언트 연결 종료")
    except Exception as e:
        print("❌ 에러 발생:", e)
        traceback.print_exc()
        await websocket.send_text("❌ 서버에서 에러가 발생했습니다.")
