"""Compare the frozen river-range baseline with and without range weighting."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "bots" / "river_range_baseline" / "bot.py"
HERO = "river_range_baseline"

SPARRING_BOT = """
import random

import eval7


def decide(state):
    if state.get("type") == "warmup":
        return {}

    if state["street"] == "river":
        board = [eval7.Card(card) for card in state["community_cards"]]
        cards = [eval7.Card(card) for card in state["your_cards"]]
        score = eval7.evaluate(board + cards)
        category = eval7.handtype(score)
        value = score > eval7.evaluate(board) and category in (
            "Straight", "Flush", "Full House", "Quads", "Straight Flush"
        )
        bluff = (
            BLUFF and state["can_check"] and category in ("High Card", "Pair")
            and random.random() < 0.75
        )
        if value or bluff:
            return {
                "action": "raise",
                "amount": max(state["min_raise_to"], state["current_bet"] + state["pot"]),
            }
        return {"action": "check" if state["can_check"] else "fold"}

    if state["can_check"]:
        return {"action": "check"}
    return {"action": "call" if 2 * state["amount_owed"] <= state["pot"] else "fold"}
"""


def play_match(hero, opponent, seed, swapped, hands, log):
    bots = (opponent, hero) if swapped else (hero, opponent)
    command = [
        sys.executable, "-B", str(ROOT / "play.py"), *map(str, bots),
        "--mode", "cash", "--hands", str(hands), "--blinds", "25/50",
        "--seed", str(seed), "--log", str(log),
    ]
    run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(run.stdout + run.stderr)

    net = completed = 0
    decisions = {}
    with log.open(encoding="utf-8") as events:
        for line in events:
            event = json.loads(line)
            if event["type"] == "hand_end":
                completed += 1
                net += event["deltas"][HERO]
            if event["type"] != "decision" or event["bot"] != HERO:
                continue
            state = event["state"]
            if event["street"] == "river" and state["amount_owed"]:
                key = (seed, swapped, event["hand"], json.dumps(state["action_log"], sort_keys=True))
                decisions[key] = (event["applied"]["action"], state, event["debug"]["equity"])

    if completed != hands:
        raise RuntimeError(f"Expected {hands} hands, played {completed}.")
    return net, decisions


def report_changes(unweighted, weighted):
    shared = unweighted.keys() & weighted.keys()
    changed = [key for key in shared if unweighted[key][0] != weighted[key][0]]
    print(f"  Same betting history at {len(shared)} river decisions, {len(changed)} action changes")
    for key in sorted(changed, key=lambda item: (-weighted[item][1]["amount_owed"], item))[:5]:
        old_action, _, old_equity = unweighted[key]
        new_action, state, new_equity = weighted[key]
        seed, swapped, hand, _ = key
        print(
            f"  Seed {seed}, {'swapped' if swapped else 'original'} seats, hand {hand + 1}, "
            f"{' '.join(state['your_cards'])} on {' '.join(state['community_cards'])}, "
            f"owed {state['amount_owed']} into pot {state['pot']}: "
            f"{old_action} ({old_equity:.1%}) -> {new_action} ({new_equity:.1%})"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 211, 307])
    parser.add_argument("--hands", type=int, default=20, help="Cash hands per seed and seat order")
    args = parser.parse_args()
    if args.hands < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use a positive hand count and distinct seeds.")

    hands = 2 * len(args.seeds) * args.hands
    with TemporaryDirectory(prefix="river-ranges-") as directory:
        scratch = Path(directory)
        unweighted = scratch / HERO
        unweighted.mkdir()
        # Override only the range weights in a temporary copy. The folder
        # name stays identical so play.py derives the same bot seed.
        (unweighted / "bot.py").write_text(
            BASELINE.read_text(encoding="utf-8")
            + "\nOpponent.river_range_weights = lambda self, game_state, hand_scores: None\n",
            encoding="utf-8",
        )
        log = scratch / "match.jsonl"

        for name, bluff in (("river_value", False), ("river_bluffer", True)):
            opponent = scratch / name
            opponent.mkdir()
            (opponent / "bot.py").write_text(
                f"BLUFF = {bluff!r}\n" + SPARRING_BOT, encoding="utf-8"
            )
            print(f"{name}, {hands} hands per variant", flush=True)
            results = []
            for label, hero in (("Unweighted", unweighted), ("Weighted", BASELINE.parent)):
                net = 0
                decisions = {}
                for seed in args.seeds:
                    for swapped in (False, True):
                        delta, observed = play_match(hero, opponent, seed, swapped, args.hands, log)
                        net += delta
                        decisions.update(observed)
                results.append((net, decisions))
                print(
                    f"  {label:10} {net:+,} chips ({100 * net / hands:+,.0f} chips/100 hands), "
                    f"{len(decisions)} river decisions facing a bet",
                    flush=True,
                )
            (old_net, old_decisions), (new_net, new_decisions) = results
            print(f"  Weighted minus unweighted: {100 * (new_net - old_net) / hands:+,.0f} chips/100 hands")
            report_changes(old_decisions, new_decisions)


if __name__ == "__main__":
    main()
