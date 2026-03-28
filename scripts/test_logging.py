import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch, logger
from config import get_config
from model import AlphaCodeModel
from mcts.search import MCTSSearch
from data.dataset import load_problems
from data.prompt_templates import format_prompt, extract_code
from evaluation.test_case_loader import load_test_cases
from evaluation.code_executor import CodeExecutor
from evaluation.reward import compute_reward

config = get_config()
model = AlphaCodeModel(config.model)
model.to_device("mps")
model.backbone.eval()
mcts = MCTSSearch(model, config.mcts)
executor = CodeExecutor(timeout=10)
problems = load_problems()
logger.banner()
logger.eval_start(3)
for p in problems[:3]:
    prompt = format_prompt(p.prompt)
    ids = model.tokenizer.encode(prompt, add_special_tokens=True)
    tc = load_test_cases(p.answer)
    with torch.no_grad():
        raw, _ = mcts.generate_solution(ids, use_mcts=False)
    code = extract_code(prompt + raw)
    r = executor.evaluate(code, tc)
    rw = compute_reward(r)
    logger.eval_problem(p.title, p.rating, r.passed, r.total, rw == 1.0, code=code)
logger.eval_summary(0, 3)
