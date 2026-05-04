"""
Hanoi Evaluator

하노이 탑 퍼즐 평가
"""

import logging
import re
import time
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


class HanoiEvaluator(BaseEvaluator):
    """
    Hanoi 퍼즐 평가자
    
    답변 형식: (disk, from, to) 튜플
    """
    
    SYSTEM_PROMPT = """### Instructions
You are an expert at Tower of Hanoi puzzles.
Always end with exactly one Answer: line containing a 3-integer tuple.

### Answer format by question type
- Move query (which disk moves, from where, to where) → Answer: (disk, from_peg, to_peg)
- Disk location (where is disk d after k moves) → Answer: (disk, peg, peg)
- Three disk locations (where are disk d1, d2, d3) → Answer: (peg_of_d1, peg_of_d2, peg_of_d3)
- Disks on a peg (list the disks on peg X) → Answer: (disk1, disk2, peg_X)
- Move count for disk k → Answer: (k, count, count)
- Corrupted board (one disk misplaced, find+fix, then continue) → Answer: (top_peg0, top_peg1, top_peg2)

### Rules
1. Follow the peg labels given in the user message.
2. Explain your reasoning, then write exactly one Answer: line at the end.
3. Do not add any text after the Answer: line.

### Output format
Answer: (a, b, c)"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
하노이 탑 퍼즐 전문가입니다.
반드시 마지막 줄을 Answer: 형식으로 작성하세요.

### 문제 유형별 답변 형식
- 이동 문제 (어떤 원판이 어디서 어디로) → Answer: (원반, 출발기둥, 도착기둥)
- 원판 위치 (k번째 이동 후 원판 d의 위치) → Answer: (원판, 기둥, 기둥)
- 세 원판 위치 (원판 d1, d2, d3의 위치) → Answer: (d1의기둥, d2의기둥, d3의기둥)
- 기둥의 원판 목록 → Answer: (원판1, 원판2, 기둥번호)
- 원판 k의 이동 횟수 → Answer: (k, 횟수, 횟수)
- 오류 보드 (1개 오배치, 찾아서 교정 후 계속) → Answer: (기둥0최상단, 기둥1최상단, 기둥2최상단)

### 규칙
1. 사용자 메시지의 기둥 번호를 따르세요.
2. 풀이 과정을 설명한 후 마지막에 Answer: 줄을 하나만 작성하세요.
3. Answer: 줄 이후에는 텍스트를 추가하지 마세요.

### 출력 형식
Answer: (a, b, c)"""

    def _is_korean(self, puzzle: Optional[Dict] = None) -> bool:
        """Prefer task_name (e.g. …_ko_easy); else infer from expected answer."""
        task = getattr(self, "_task_name", None) or ""
        hint = locale_from_task_name(task)
        if hint is not None:
            return hint
        if puzzle is not None:
            expected = puzzle.get("answer", "")
            return bool(re.search(r"[가-힣]", str(expected)))
        return False

    def _get_system_prompt(self, puzzle: Dict) -> str:
        if self._is_korean(puzzle):
            return self.KOREAN_SYSTEM_PROMPT
        return self.SYSTEM_PROMPT

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient",
    ) -> EvaluationResult:
        system_prompt = self._get_system_prompt(puzzle)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": puzzle["question"]},
        ]
        start = time.time()
        try:
            response, usage = llm_client.generate(messages)
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, response, latency, usage)
        except Exception as e:
            latency = (time.time() - start) * 1000
            return self._process_response(puzzle, "", latency, {"error": str(e)})

    async def _evaluate_async(
        self,
        puzzles: List[Dict[str, Any]],
        llm_client: "BaseLLMClient",
        verbose: bool = True,
        max_concurrent: int = 10,
    ) -> List[EvaluationResult]:
        from ..core.base import logger

        messages_list = []
        for puzzle in puzzles:
            system_prompt = self._get_system_prompt(puzzle)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": puzzle["question"]},
            ]
            messages_list.append(messages)

        total_puzzles = len(puzzles)
        task_name = getattr(self, "_task_name", None)
        task_prefix = f"[{task_name}] " if task_name else ""

        if verbose:
            logger.info(
                f"{task_prefix}Starting async evaluation: {total_puzzles} puzzles, "
                f"max_concurrent={max_concurrent}"
            )

        start_time = time.time()

        def progress_callback(completed, total):
            if verbose:
                percentage = (completed / total) * 100
                if completed % max(1, total // 10) == 0 or completed == total:
                    logger.info(
                        f"{task_prefix}API calls progress: {completed}/{total} ({percentage:.0f}%)"
                    )

        responses = await llm_client.async_batch_generate(
            messages_list,
            max_concurrent=max_concurrent,
            progress_callback=progress_callback if verbose else None,
        )
        total_latency = (time.time() - start_time) * 1000

        if verbose:
            logger.info(
                f"{task_prefix}API calls completed: {total_puzzles}/{total_puzzles} in "
                f"{total_latency:.0f}ms ({total_latency/total_puzzles:.0f}ms per puzzle)"
            )

        results = []
        correct_count = 0
        error_count = 0

        for puzzle, (response, usage) in zip(puzzles, responses):
            latency_ms = usage.get("latency_ms", 0)
            result = self._process_response(puzzle, response, latency_ms, usage)
            if result.correct:
                correct_count += 1
            if result.error:
                error_count += 1
            results.append(result)

        if verbose:
            incorrect_count = total_puzzles - correct_count - error_count
            logger.info(
                f"Processing completed: {correct_count} correct, {incorrect_count} incorrect, "
                f"{error_count} errors"
            )

        return results

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[Tuple[int, int, int]]:
        """
        LLM 응답에서 (disk, from, to) 튜플 추출
        
        숫자 3개를 찾아서 튜플로 반환
        
        Args:
            response: LLM 응답 텍스트
            puzzle: 퍼즐 데이터 (사용하지 않음)
        """
        answer_text = self._extract_final_answer_text(response) or response

        # 패턴 1: (숫자, 숫자, 숫자) 형식
        match = re.search(r'\((\d+),\s*(\d+),\s*(\d+)\)', answer_text)
        if match:
            return (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # 패턴 2: 숫자 3개 추출
        nums = re.findall(r'\d+', answer_text)
        if len(nums) >= 3:
            return (int(nums[0]), int(nums[1]), int(nums[2]))

        # 단일 숫자 답변 보정: 역문제/총 이동 횟수 문제
        question_raw = str(puzzle.get("question", ""))
        question = question_raw.lower()
        if len(nums) == 1:
            v = int(nums[0])

            # How many disks total? (inverse_find_n) -> (n, n, n)
            # Exclude "top disk per peg" questions which also ask about disks per peg
            is_inverse_en = (
                "how many disks" in question
                and "top" not in question
                and "top_of_peg" not in question
            )
            is_inverse_ko = (
                "원판이 몇 개" in question_raw
                and "최상단" not in question_raw
            )
            if is_inverse_en or is_inverse_ko:
                return (v, v, v)

            # How many times does Disk k move? -> (k, t, t)
            m = re.search(r"disk\s*(\d+)", question, re.IGNORECASE)
            if "how many times" in question and m:
                k = int(m.group(1))
                return (k, v, v)

            # Korean variant: "원반 k" + "몇 번"
            m_ko = re.search(r"원반\s*(\d+)", question_raw)
            if "몇 번" in question_raw and m_ko:
                k = int(m_ko.group(1))
                return (k, v, v)
        
        return None
    
    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[Tuple[int, int, int]]
    ) -> Tuple[bool, float]:
        """
        답변 확인
        
        Returns:
            (is_correct, partial_score) 튜플
        """
        if predicted is None:
            return False, 0.0
        
        # expected가 문자열일 수 있으므로 파싱
        if isinstance(expected, str):
            expected = self._parse_answer(expected, {})
        
        correct = predicted == expected
        return correct, 1.0 if correct else 0.0
