import sys
import time
from pathlib import Path

import requests

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evaluation.model.remote import RemoteLLMClient, accumulate_openai_chat_sse_lines

URL = "https://tremendously-bureaucratic-alda.ngrok-free.dev"

system_prompt = """당신은 뱃사공 운항 문제를 정확히 해결하는 전문가입니다.

### 규칙
1. 주어진 운항 규정을 모두 고려하여 단계별로 분석하세요.
2. 속도 제한, 의무 휴식, 화물 규정을 모두 적용하여 계산하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 최종 답 형식
마지막에 $\\boxed{N시간 M분}$ 형식으로 정답을 표시하세요.
"""

content = """
수로 안내원 오씨는 이 강에서만 9년을 일한 베테랑으로, 총 길이 206km의 상류 지역에 물품을 운송하는 임무를 맡았다. 그는 오전 8시에 12kg짜리 의료품 키트 10개와 15kg짜리 식수통 5개를 싣고 출발했다. 배는 정수(靜水)에서 시속 22km로 이동 가능하다.
이 강에는 시속 7km의 물살이 있는데, B구역에서는 순류(실효 속력 = 배 속력 + 유속), A구역에서는 역류(실효 속력 = 배 속력 - 유속)이다.
첫 23km는 A구역(제한속도 17km/h), 이후 B구역(제한속도 13km/h)이다. 제한 속도는 유속 반영 후 실효 속력에 적용된다.
안전 중량 기준(400kg) 초과 시, A구역 제한속도는 21% 감소하고 B구역 제한속도는 33% 감소한다.
오전 9시부터 오후 2시까지는 혼잡시간대로, 모든 구역의 제한속도가 추가로 41% 감소한다. 이 감속은 화물 규정 적용 후 제한속도에 추가 적용된다.
60km 지점의 중간 기착지에서 식수통 3개를 하역한다. 하역 후 잔여 화물 무게에 따라 화물 규정이 재적용된다.
연속 101분 이상 운항할 수 없으며, 휴게 지점(매 12km)에서만 쉴 수 있다. 기본 휴식 시간은 44분이다. 단, 누적 피로 규정에 따라 매 휴식마다 20분씩 추가된다(첫째 44분, 둘째 64분, 셋째 84분, …).
이 모든 조건을 준수하여 최종 목적지까지 도착했을 때, 의무 휴식을 포함한 총 소요 시간은 몇 시간 몇 분입니까? (분은 소숫점 첫째 자리에서 반올림)\
"""

# hard 68시간 24분

data = {
    "model": "Qwen/Qwen3.5-9B",
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "안녕"}
    ],
    "temperature": 1.0,     
    "max_tokens": 72768, 
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 1.5,
    "repetition_penalty": 1.0,
    "stream": True,
    # vLLM 서버 설정에서 --reasoning-parser qwen3를 사용하므로
    # extra_body에 넣어주는 것이 가장 정확합니다.
    "extra_body": {
        "enable_thinking": True,
        "min_p": 0.0 
    }
}

print(f"[URL] {URL}/v1/chat/completions")
print(f"[Model] {data['model']}")
# 수정: data['extra_body']에서 값을 가져오도록 변경 (KeyError 방지)
print(f"[Thinking] {data['extra_body'].get('enable_thinking', False)}")
print(f"[Prompt] {data['messages']}")
print("-" * 50)

start = time.time()
try:
    with requests.post(
        f"{URL}/v1/chat/completions",
        json=data,
        timeout=600,
        stream=True,
    ) as resp:
        print(f"[HTTP Status] {resp.status_code}")

        if resp.status_code != 200:
            print("[Body]")
            print(resp.text[:4000])
            raise SystemExit(1)

        content, thinking, usage = accumulate_openai_chat_sse_lines(
            resp.iter_lines(decode_unicode=True)
        )
        content, thinking = RemoteLLMClient._finalize_stream_texts(content, thinking)
        print(f"[Response Time] {time.time() - start:.2f}s (full stream)")

    thinking = (thinking or "").strip()
    content = (content or "").strip()

    if thinking:
        print(f"\n[Reasoning]\n{thinking}")
    print(f"\n[Answer]\n{content}")

    if usage:
        print(f"\n[Usage] {usage}")

except requests.exceptions.Timeout:
    print(f"[Error] Timeout: {URL}")
except requests.exceptions.ConnectionError:
    print(f"[Error] Connection failed: {URL} — Check Colab server/ngrok URL.")
except Exception as e:
    print(f"[Error] {e}")

# python scripts/calltest_qwen.py