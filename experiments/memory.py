"""Few-shot prompt construction and solving with retrieved exemplars.
`solve` returns (answer, confidence, generated text); every experiment calls it."""
import re

_ANS = re.compile(r"ANSWER:\s*(-?\d+)")
_CONF = re.compile(r"CONFIDENCE:\s*(\d{1,3})")


def build_prompt(problem, exemplars):
    shots = ""
    for ex in exemplars:  # show the WORKED solution (procedure), not just the answer
        shots += f"Solved example:\nProblem: {ex['problem']}\nSolution: {ex['sol']}\n\n"
    return (shots + "Solve the new problem, using the solved examples as guidance.\n"
            f"Problem: {problem}\nEnd with EXACTLY two lines:\n"
            "ANSWER: <integer>\nCONFIDENCE: <0-100>")


def solve(be, problem, exemplars):
    txt = be.gen(build_prompt(problem, exemplars))
    a, c = _ANS.search(txt), _CONF.search(txt)
    return (int(a.group(1)) if a else None,
            max(0, min(100, int(c.group(1)) if c else 50)) / 100.0, txt)
