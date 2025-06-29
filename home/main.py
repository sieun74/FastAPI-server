from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import httpx
import traceback

app = FastAPI()

# Groq API 설정
GROQ_API_KEY = 'grop_api_key'
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
