"""Play the tournament bot against reference policies in OpenSpiel Hold'em."""

import argparse
import importlib.util
import random
import re
from collections import deque
from pathlib import Path
from time import perf_counter

import pyspiel


ROOT = Path(__file__).resolve().parents[2]
STREETS = ("preflop", "flop", "turn", "river")
CARDS = re.compile(r"[2-9TJQKA][cdhs]")
PRIVATE = re.compile(r"\[Private: ([^\]]+)\]")
FOLD, CHECK_CALL = 0, 1
SMALL_BLIND, BIG_BLIND = 25, 50
DECISION_LIMIT_MS = 2000


def load_bot(path):
    spec = importlib.util.spec_from_file_location("tournament_entry", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load a Python bot from {path}")
    bot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bot)
    return bot


def private_cards(state, seat):
    match = PRIVATE.search(state.observation_string(seat))
    cards = CARDS.findall(match.group(1)) if match else []
    if len(cards) != 2:
        raise RuntimeError("Unexpected OpenSpiel private-card observation.")
    return cards


def encode_action(reply, legal, prior_commitment):
    action = reply["action"]
    if action == "fold":
        return FOLD if FOLD in legal else CHECK_CALL
    if action in ("call", "check"):
        return CHECK_CALL
    if action not in ("raise", "all_in"):
        raise ValueError(f"Unknown bot action {action!r}")
    raises = [move for move in legal if move > CHECK_CALL]
    if not raises:
        return CHECK_CALL
    if action == "all_in":
        return raises[-1]
    target = prior_commitment + int(reply["amount"])
    return min(max(target, raises[0]), raises[-1])


def play_hand(game, bot, opponent, hero, stack, hand_num, history, rng):
    state = game.new_initial_state()
    names = ["opponent", "opponent"]
    names[hero] = "72_offsuit"
    board, action_log = [], []
    contributions = [BIG_BLIND, SMALL_BLIND]
    prior = [0, 0]
    round_index = 0
    deals = 0
    slowest_ms = 0.0

    def record(seat, action, amount):
        entry = dict(seat=seat, bot_id=names[seat], action=action, amount=amount,
                     street=STREETS[round_index])
        action_log.append(entry)
        history.append(dict(entry, hand_num=hand_num))

    record(1, "small_blind", SMALL_BLIND)
    record(0, "big_blind", BIG_BLIND)
    while not state.is_terminal():
        seat = state.current_player()
        if state.is_chance_node():
            outcomes, weights = zip(*state.chance_outcomes())
            move = rng.choices(outcomes, weights=weights, k=1)[0]
            if deals >= 4:
                label = state.action_to_string(seat, move)
                card = CARDS.search(label)
                if not card:
                    raise RuntimeError(f"Unexpected OpenSpiel deal action {label!r}")
                board.append(card.group())
            deals += 1
            state.apply_action(move)
            continue

        street_index = max(0, len(board) - 2)
        if street_index != round_index:
            prior = contributions.copy()
            round_index = street_index
        committed = [total - before for total, before in zip(contributions, prior)]
        current_bet = max(committed)
        owed = current_bet - committed[seat]
        stacks = [stack - total for total in contributions]
        legal = state.legal_actions()
        if seat == hero:
            min_total = next((move for move in legal if move > CHECK_CALL), stack)
            players = [
                dict(seat=p, bot_id=names[p], stack=stacks[p],
                     state="active" if stacks[p] else "all_in",
                     is_folded=False, is_all_in=not stacks[p], bet_this_street=committed[p],
                     hole_cards=None)
                for p in range(2)
            ]
            request = dict(
                type="action_request", street=STREETS[round_index], hand_num=hand_num,
                hand_id=f"openspiel-{hand_num}", seat_to_act=seat, your_bot_id=names[seat],
                your_cards=private_cards(state, seat), community_cards=list(board),
                pot=sum(contributions), current_bet=current_bet, amount_owed=owed,
                can_check=owed == 0, min_raise_to=min_total - prior[seat],
                your_bet_this_street=committed[seat], your_stack=stacks[seat],
                button_seat=1, sb_seat=1, bb_seat=0,
                blinds=dict(sb=SMALL_BLIND, bb=BIG_BLIND, ante=0),
                players=players, action_log=list(action_log), match_action_log=list(history),
                time_limit_ms=DECISION_LIMIT_MS,
            )
            start = perf_counter()
            reply = bot.decide(request)
            slowest_ms = max(slowest_ms, (perf_counter() - start) * 1000)
            move = encode_action(reply, legal, prior[seat])
        else:
            move = opponent.step(state)

        if move not in legal:
            raise RuntimeError(f"Illegal OpenSpiel action {move} at hand {hand_num + 1}")
        if move == FOLD:
            record(seat, "fold", 0)
        elif move == CHECK_CALL:
            paid = min(owed, stacks[seat])
            contributions[seat] += paid
            record(seat, "call" if owed else "check", paid)
        else:
            contributions[seat] = move
            record(seat, "all_in" if move == stack else "raise", move - prior[seat])
        state.apply_action(move)

    if action_log[-1]["action"] != "fold":
        for seat in range(2):
            history.append(dict(
                seat=seat, bot_id=names[seat], hand_num=hand_num, street="showdown",
                action="show", amount=0, cards=private_cards(state, seat),
            ))
    return state.player_return(hero), slowest_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot", type=Path, default=ROOT / "my_bots/72_offsuit/bot.py")
    parser.add_argument("--opponent", choices=("random", "check-call", "fold"), default="check-call")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--hands", type=int, default=50, help="Cash hands per seat")
    parser.add_argument("--stack", type=int, default=10_000)
    args = parser.parse_args()
    if args.hands < 1 or args.stack < BIG_BLIND * 2 or not 0 <= args.seed < 2**31:
        parser.error("Use positive hands, a stack of at least 100, and a seed in [0, 2**31).")
    if "universal_poker" not in pyspiel.registered_names():
        parser.error("OpenSpiel must be built with ACPC poker support.")

    params = {
        "betting": "nolimit", "numPlayers": 2, "numRounds": 4,
        "blind": f"{BIG_BLIND} {SMALL_BLIND}", "firstPlayer": "2 1 1 1",
        "numSuits": 4, "numRanks": 13, "numHoleCards": 2, "numBoardCards": "0 3 1 1",
        "stack": f"{args.stack} {args.stack}", "bettingAbstraction": "fullgame",
    }
    game = pyspiel.load_game("universal_poker", params)
    net = slowest = 0
    for hero in range(2):
        bot = load_bot(args.bot)
        random.seed(args.seed)
        bot.decide({"type": "warmup"})
        if args.opponent == "random":
            opponent = pyspiel.make_uniform_random_bot(1 - hero, args.seed)
        else:
            order = [CHECK_CALL, FOLD] if args.opponent == "check-call" else [FOLD, CHECK_CALL]
            opponent = pyspiel.make_policy_bot(game, 1 - hero, args.seed, pyspiel.PreferredActionPolicy(order))
        history = deque(maxlen=200)
        results = [
            play_hand(game, bot, opponent, hero, args.stack, hand, history,
                      random.Random(f"{args.seed}:{hand}"))
            for hand in range(args.hands)
        ]
        chips = sum(result for result, _ in results)
        net += chips
        slowest = max(slowest, max(ms for _, ms in results))
        print(f"{'Small' if hero else 'Big'} blind: {chips:+,.0f} chips over {args.hands} hands")
    print(f"Paired result vs {args.opponent}: {net:+,.0f} chips, "
          f"{100 * net / (2 * args.hands * BIG_BLIND):+,.1f} bb/100, max decision {slowest:.1f} ms")


if __name__ == "__main__":
    main()
