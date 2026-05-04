#!/usr/bin/env bash
# gen_data.sh와 동일 순서로 데이터를 생성한 뒤,
# data/jsonl/*.jsonl(이미 *_easy 등으로 끝나는 파일 제외)을 난이도별로 쪼개고,
# 합본(foo_en.jsonl)은 삭제해 task당 *_easy|medium|hard.jsonl 세 개만 남긴다.
# 난이도당 문항 수: PER_DIFF(기본 100). 총 300문항 생성기는 PER_DIFF*3 으로 호출한다.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$PROJECT_ROOT"

PER_DIFF="${PER_DIFF:-100}"
TOTAL=$((PER_DIFF * 3))

echo "============================================"
echo "Logical Puzzles — generate + split by difficulty"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "PER_DIFF=$PER_DIFF (TOTAL split generators: $TOTAL)"
echo "============================================"
echo ""

run_py() {
  echo ""
  echo "$1"
  shift
  python "$@"
}

# --- generation (mirror run/generate/gen_data.sh; --num semantics per script) ---

# # run_py "array_formula_en..." generation/array_formula_en.py --num "$PER_DIFF"
# run_py "array_formula_ko..." generation/array_formula_ko.py --num "$PER_DIFF"

# # run_py "cipher_en..." generation/cipher_en.py --num "$PER_DIFF"
# run_py "cipher_ko..." generation/cipher_ko.py --num "$PER_DIFF"

# # run_py "ferryman_en..." generation/ferryman_en.py --num "$PER_DIFF"
# run_py "ferryman_ko..." generation/ferryman_ko.py --num "$PER_DIFF"

# run_py "hanoi_en..." generation/hanoi_en.py --num "$PER_DIFF"
# run_py "hanoi_ko..." generation/hanoi_ko.py --num "$PER_DIFF"

# # run_py "causal_dag_en..." generation/causal_dag_en.py --num "$TOTAL"
# run_py "causal_dag_ko..." generation/causal_dag_ko.py --num "$TOTAL"

# # run_py "logic_grid_en..." generation/logic_grid_en.py --num-samples "$TOTAL"
# run_py "logic_grid_ko..." generation/logic_grid_ko.py --num-samples "$TOTAL"

# # run_py "sat_puzzle_en..." generation/sat_puzzle_en.py --num-samples "$TOTAL"
# run_py "sat_puzzle_ko..." generation/sat_puzzle_ko.py --num-samples "$TOTAL"

# run_py "inequality_en..." generation/inequality_en.py --num "$TOTAL"
# run_py "inequality_ko..." generation/inequality_ko.py --num "$TOTAL"

# run_py "minesweeper_en..." generation/minesweeper_en.py --num "$TOTAL"
# run_py "minesweeper_ko..." generation/minesweeper_ko.py --num "$TOTAL"

# run_py "number_baseball_en..." generation/number_baseball_en.py --num "$TOTAL"
# run_py "number_baseball_ko..." generation/number_baseball_ko.py --num "$TOTAL"

# run_py "sudoku_en..." generation/sudoku_en.py --num "$TOTAL"
# run_py "sudoku_ko..." generation/sudoku_ko.py --num "$TOTAL"

# run_py "yacht_dice_en..." generation/yacht_dice_en.py --num "$TOTAL"
# run_py "yacht_dice_ko..." generation/yacht_dice_ko.py --num "$TOTAL"

# run_py "cryptarithmetic_en..." generation/cryptarithmetic_en.py --num "$TOTAL"
# run_py "cryptarithmetic_ko..." generation/cryptarithmetic_ko.py --num "$TOTAL"



# run_py "kinship..." generation/kinship.py --num "$PER_DIFF"
# run_py "kinship_vision..." generation/kinship_vision.py --num "$PER_DIFF"

echo ""
echo "============================================"
echo "Splitting combined JSONL -> *_easy|medium|hard.jsonl"
echo "============================================"
python scripts/gen/split_jsonl_by_difficulty.py --jsonl-dir "$PROJECT_ROOT/data/jsonl" --max-per-difficulty "$PER_DIFF" --delete-source

echo ""
echo "Done. JSONL outputs: data/jsonl/<stem>_{easy,medium,hard}.jsonl (combined *.jsonl removed)."


# bash run/generate/gen_data_by_difficulty.sh