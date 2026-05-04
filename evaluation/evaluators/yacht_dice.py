"""
Yacht Dice Evaluator

Evaluates Yacht Dice puzzle responses.
Answer format: integer (total score or spotcheck round sum).

Features ported from logical-puzzles-me/yacht_dice:
- Config-rendered system prompts (full_house_points, bonus_threshold, etc.)
- Round-level spotcheck (medium/hard by default): select K rounds deterministically
  and compare the sum of optimally-assigned round scores instead of the total.
"""

import hashlib
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TYPE_CHECKING

from ..core.base import BaseEvaluator, EvaluationResult
from ..task_names import locale_from_task_name

if TYPE_CHECKING:
    from ..model.base import BaseLLMClient

logger = logging.getLogger(__name__)


# ============================================================
# Config (mirrors generator's YachtDiceConfig)
# ============================================================

@dataclass
class YachtDiceConfig:
    bonus_threshold: int = 63
    bonus_points: int = 35
    full_house_points: int = 25
    small_straight_points: int = 30
    large_straight_points: int = 40
    yacht_points: int = 50
    optimization_goal: Literal["maximize", "minimize"] = "maximize"


# Number of rounds revealed in the spotcheck prompt per difficulty.
# "easy" defaults to no spotcheck (full-total answer), medium/hard use spotcheck.
SPOTCHECK_K = {"easy": 0, "medium": 4, "hard": 5}


UPPER_CATEGORIES = {"Aces", "Twos", "Threes", "Fours", "Fives", "Sixes"}


def _score_category(dice: List[int], category: str, config: YachtDiceConfig) -> int:
    counts = Counter(dice)
    sorted_dice = sorted(dice)

    if category == "Aces":
        return dice.count(1) * 1
    elif category == "Twos":
        return dice.count(2) * 2
    elif category == "Threes":
        return dice.count(3) * 3
    elif category == "Fours":
        return dice.count(4) * 4
    elif category == "Fives":
        return dice.count(5) * 5
    elif category == "Sixes":
        return dice.count(6) * 6
    elif category == "Three-Of-A-Kind":
        for _, c in counts.items():
            if c >= 3:
                return sum(dice)
        return 0
    elif category == "Four-Of-A-Kind":
        for _, c in counts.items():
            if c >= 4:
                return sum(dice)
        return 0
    elif category == "Full House":
        if sorted(counts.values()) == [2, 3]:
            return config.full_house_points
        return 0
    elif category == "Small Straight":
        unique = set(sorted_dice)
        for straight in ({1, 2, 3, 4}, {2, 3, 4, 5}, {3, 4, 5, 6}):
            if straight.issubset(unique):
                return config.small_straight_points
        return 0
    elif category == "Large Straight":
        unique = set(sorted_dice)
        if unique == {1, 2, 3, 4, 5} or unique == {2, 3, 4, 5, 6}:
            return config.large_straight_points
        return 0
    elif category == "Yacht":
        if len(counts) == 1:
            return config.yacht_points
        return 0
    return 0


class YachtDiceEvaluator(BaseEvaluator):
    """
    Yacht Dice puzzle evaluator.

    Parses an integer answer. For medium/hard, appends a spotcheck instruction
    asking the model to report the sum of scores for a deterministic subset of
    rounds (based on optimal_assignment) instead of the total score.
    """

    CONFIG = YachtDiceConfig()

    SYSTEM_PROMPT = """### Instructions
You are an expert at Yacht Dice (Yahtzee-style) score assignment optimization.

### Rules
1. Follow the dice rolls, scoring categories, and point values exactly as given in the user message (including bonuses and category caps).
2. Assign each of the 12 rounds to a category at most once so the objective (usually maximize total or a stated spotcheck sum) is met optimally unless the puzzle specifies otherwise.
3. Explain your reasoning clearly, then present your final conclusion in the format below.

### Output format
Your final line must be:
Answer: [number]
"""

    KOREAN_SYSTEM_PROMPT = """### 지시사항
당신은 요트 다이스(Yacht Dice, 야추 스타일) 점수 배정 최적화 문제를 정확히 푸는 전문가입니다.

### 규칙
1. 사용자 메시지에 제시된 주사위·점수 칸·보너스 규칙을 그대로 따르세요.
2. 12라운드를 각 칸에 최대 한 번씩 배정하여, 목표(총점 또는 별도로 제시된 스팟체크 합 등)에 맞게 최적으로 배치하세요.
3. 풀이 과정을 명확히 서술한 뒤, 최종 결론을 아래 형식으로 제시하세요.

### 출력 형식
마지막 줄은 반드시 아래 형식으로 작성하세요:
Answer: [숫자]
"""

    # ========================================================================
    # Language helpers
    # ========================================================================

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

    def _render_system_prompt(self, korean: bool) -> str:
        return self.KOREAN_SYSTEM_PROMPT if korean else self.SYSTEM_PROMPT

    def _get_system_prompt(self, puzzle: Dict) -> str:
        return self._render_system_prompt(self._is_korean(puzzle))

    # ========================================================================
    # Spotcheck
    # ========================================================================

    def _spotcheck_enabled(self, puzzle: Dict) -> bool:
        difficulty = puzzle.get("difficulty", "easy")
        k = SPOTCHECK_K.get(difficulty, 0)
        if k <= 0:
            return False
        if not (puzzle.get("optimal_assignment") and puzzle.get("dice_results")):
            return False
        # Gate on optimal-assignment uniqueness: if multiple optimal assignments
        # exist, the stored one is arbitrary and spotcheck would mark legitimate
        # alternate-optimum answers as wrong. Fall back to total-score compare.
        sm = puzzle.get("step_metrics") or {}
        return bool(sm.get("is_unique_assignment"))

    def _deterministic_round_pick(self, puzzle: Dict, k: int) -> List[int]:
        """Pick K 1-indexed round numbers deterministically from puzzle id."""
        key = str(puzzle.get("id", ""))
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        # Stable pseudo-random permutation of [0..11]
        ordering = sorted(range(12), key=lambda i: digest[i % len(digest)] * 13 + i)
        selected = sorted(ordering[:k])
        return [i + 1 for i in selected]

    def _compute_spotcheck(self, puzzle: Dict) -> Optional[Dict]:
        """Compute spotcheck metadata: picked rounds (1-indexed) and expected sum."""
        if not self._spotcheck_enabled(puzzle):
            return None

        difficulty = puzzle.get("difficulty", "easy")
        k = SPOTCHECK_K.get(difficulty, 0)
        dice_results = puzzle.get("dice_results") or []
        assignment = puzzle.get("optimal_assignment") or {}

        if len(dice_results) != 12 or len(assignment) != 12:
            return None

        rounds_1b = self._deterministic_round_pick(puzzle, min(k, 12))

        # Normalize assignment keys (may be str or int from JSONL).
        norm_assignment: Dict[int, str] = {}
        for key, cat in assignment.items():
            try:
                norm_assignment[int(key)] = cat
            except (TypeError, ValueError):
                continue

        expected_sum = 0
        for r1 in rounds_1b:
            idx0 = r1 - 1
            cat = norm_assignment.get(idx0)
            if cat is None:
                return None
            expected_sum += _score_category(dice_results[idx0], cat, self.CONFIG)

        return {"rounds": rounds_1b, "expected_sum": expected_sum}

    def _build_spotcheck_suffix(self, spotcheck: Dict, korean: bool) -> str:
        rounds = spotcheck["rounds"]
        round_list = ", ".join(str(r) for r in rounds)
        if korean:
            return (
                "\n\n중요: 총점을 제시하지 마세요. 대신:\n"
                "1. 12라운드 모두에 대해 최적 카테고리 배정을 찾으세요\n"
                f"2. 다음 라운드만 해당: {round_list}\n"
                "3. 이 라운드들이 배정된 카테고리에서 얻는 점수를 각각 계산하세요\n"
                "4. 해당 점수들만 합산하세요\n\n"
                f"중요: 답은 라운드 {round_list}의 점수 합계만이어야 합니다.\n"
                f"Answer: [라운드 {round_list}의 점수 합계, 정수]"
            )
        else:
            return (
                "\n\nIMPORTANT: Do NOT provide the total score. Instead:\n"
                "1. Find the optimal category assignment for all 12 rounds\n"
                f"2. For ONLY these rounds: {round_list}\n"
                "3. Calculate the score each of these rounds earns with its assigned category\n"
                "4. Sum ONLY those scores\n\n"
                f"CRITICAL: Your answer must be the SUM of scores for rounds {round_list} ONLY.\n"
                f"Answer: [sum of scores for rounds {round_list} only, as integer]"
            )

    def _prepare_puzzle_for_eval(self, puzzle: Dict) -> Tuple[Dict, str]:
        """Attach transient spotcheck metadata and build user content."""
        puzzle = dict(puzzle)
        user_content = puzzle.get("question", "")

        spotcheck = self._compute_spotcheck(puzzle)
        if spotcheck is not None:
            korean = self._is_korean(puzzle)
            user_content = user_content + self._build_spotcheck_suffix(spotcheck, korean)
            puzzle["_spotcheck_rounds"] = spotcheck["rounds"]
            puzzle["_spotcheck_expected"] = spotcheck["expected_sum"]
        return puzzle, user_content

    # ========================================================================
    # Overridden evaluation entry points
    # ========================================================================

    def _evaluate_single(
        self,
        puzzle: Dict[str, Any],
        llm_client: "BaseLLMClient",
    ) -> EvaluationResult:
        puzzle, user_content = self._prepare_puzzle_for_eval(puzzle)
        messages = [
            {"role": "system", "content": self._get_system_prompt(puzzle)},
            {"role": "user", "content": user_content},
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
        enriched_puzzles: List[Dict[str, Any]] = []
        messages_list: List[List[Dict[str, str]]] = []
        for puzzle in puzzles:
            enriched, user_content = self._prepare_puzzle_for_eval(puzzle)
            enriched_puzzles.append(enriched)
            messages_list.append([
                {"role": "system", "content": self._get_system_prompt(enriched)},
                {"role": "user", "content": user_content},
            ])

        total_puzzles = len(enriched_puzzles)
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

        results: List[EvaluationResult] = []
        correct_count = 0
        error_count = 0

        for puzzle, (response, usage) in zip(enriched_puzzles, responses):
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

    # ========================================================================
    # Response processing (spotcheck-aware)
    # ========================================================================

    def _process_response(
        self,
        puzzle: Dict[str, Any],
        response: str,
        latency_ms: float,
        usage: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        usage = usage or {}

        if "error" in usage:
            return self._create_error_result(
                puzzle, response if response else "", latency_ms, usage["error"]
            )

        try:
            predicted = self._parse_answer(response, puzzle)
            spotcheck_expected = puzzle.get("_spotcheck_expected")

            if spotcheck_expected is not None:
                correct = predicted is not None and int(predicted) == int(spotcheck_expected)
                return EvaluationResult(
                    puzzle_id=puzzle["id"],
                    difficulty=puzzle.get("difficulty", "Unknown"),
                    correct=correct,
                    partial_score=1.0 if correct else 0.0,
                    expected=str(spotcheck_expected),
                    predicted=predicted,
                    raw_response=response,
                    latency_ms=latency_ms,
                    thinking_content=usage.get("thinking_content", "") if isinstance(usage, dict) else "",
                )
            else:
                correct, partial_score = self._check_answer(puzzle["answer"], predicted)
                return EvaluationResult(
                    puzzle_id=puzzle["id"],
                    difficulty=puzzle.get("difficulty", "Unknown"),
                    correct=correct,
                    partial_score=partial_score,
                    expected=puzzle["answer"],
                    predicted=predicted,
                    raw_response=response,
                    latency_ms=latency_ms,
                    thinking_content=usage.get("thinking_content", "") if isinstance(usage, dict) else "",
                )
        except Exception as e:
            return self._create_error_result(puzzle, response, latency_ms, str(e))

    # ========================================================================
    # Answer parsing / checking
    # ========================================================================

    def _parse_answer(self, response: str, puzzle: Dict) -> Optional[int]:
        """Extract integer answer from LLM response (multi-step fallback)."""
        response = self._strip_code_fences(response).strip()
        answer_text = self._extract_final_answer_text(response, allow_boxed_fallback=False) or response

        # Priority 1: "Answer:" pattern
        answer_matches = re.findall(
            r'(?:Answer|Output|Final\s*Answer)\s*[:\s]*(\d+)',
            answer_text, re.IGNORECASE
        )
        if answer_matches:
            return int(answer_matches[-1])

        # Priority 2: Total/sum patterns
        total_patterns = [
            r'[Tt]otal[:\s]*[=\s]*(\d+)',
            r'[Ss]um[:\s]*[=\s]*(\d+)',
        ]
        for pattern in total_patterns:
            matches = re.findall(pattern, answer_text)
            if matches:
                return int(matches[-1])

        # Priority 3: last number in last 5 lines (largest on line)
        lines = answer_text.strip().split('\n')
        for line in reversed(lines[-5:]):
            nums = re.findall(r'\b(\d+)\b', line.strip())
            if nums:
                return int(max(nums, key=int))

        # Priority 4: last number anywhere
        all_nums = re.findall(r'\b(\d+)\b', answer_text)
        if all_nums:
            return int(all_nums[-1])

        return None

    def _check_answer(
        self,
        expected: Any,
        predicted: Optional[int],
    ) -> Tuple[bool, float]:
        if predicted is None:
            return False, 0.0
        try:
            expected_num = int(expected)
        except (ValueError, TypeError):
            return False, 0.0
        correct = int(predicted) == expected_num
        return correct, 1.0 if correct else 0.0
