"""Run a small Slumbot benchmark with the tournament entry offline."""

import argparse
import json
import re
import sys
import time
from collections import Counter, deque
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pokercore.botproc import BotProcess


HERO, OPPONENT = "72_offsuit", "slumbot"
STACK, BIG_BLIND = 20_000, 100
STREETS = ("preflop", "flop", "turn", "river")
API = "https://slumbot.com/slumbot/api/"
BETTING = re.compile(r"b[0-9]+|[kcf]")


class SlumbotSession:
    def __init__(self):
        self.token = None
        self.next_request = 0.0

    def request(self, endpoint, **payload):
        if self.token:
            payload["token"] = self.token
        time.sleep(max(0, self.next_request - time.monotonic()))
        request = Request(
            API + endpoint, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        # POSTs advance the game. A failed response must not be retried blindly.
        with urlopen(request, timeout=20) as response:
            result = json.load(response)
        self.next_request = time.monotonic() + 0.5
        if "error_msg" in result:
            raise RuntimeError(f"Slumbot rejected {endpoint}: {result['error_msg']}")
        self.token = result.get("token", self.token)
        if not self.token:
            raise RuntimeError("Slumbot did not provide a session token.")
        return result


def parse_history(sequence, hero, hand_num):
    """Rebuild contributions, next seat, and the kit's public action log."""
    names = [HERO if seat == hero else OPPONENT for seat in range(2)]
    stacks = [STACK - BIG_BLIND, STACK - BIG_BLIND // 2]
    committed = [BIG_BLIND, BIG_BLIND // 2]
    log = []
    street_index = 0

    def record(seat, action, amount):
        log.append(dict(hand_num=hand_num, seat=seat, bot_id=names[seat],
                        street=STREETS[street_index], action=action, amount=amount))

    record(1, "small_blind", BIG_BLIND // 2)
    record(0, "big_blind", BIG_BLIND)
    rounds = sequence.split("/")
    if len(rounds) > len(STREETS):
        raise ValueError(f"Too many streets in {sequence!r}")
    folded = None
    for street_index, street_actions in enumerate(rounds):
        if street_index:
            committed = [0, 0]
        seat = 0 if street_index else 1
        increment = BIG_BLIND
        actions = BETTING.findall(street_actions)
        if "".join(actions) != street_actions:
            raise ValueError(f"Unexpected betting history {sequence!r}")
        for action in actions:
            current_bet = max(committed)
            if action.startswith("b"):
                target = int(action[1:])
                added = target - committed[seat]
                if target <= current_bet or not 0 < added <= stacks[seat]:
                    raise ValueError(f"Invalid raise in {sequence!r}")
                stacks[seat] -= added
                committed[seat] = target
                increment = max(increment, target - current_bet)
                record(seat, "all_in" if not stacks[seat] else "raise", target)
            elif action == "c":
                paid = min(current_bet - committed[seat], stacks[seat])
                if paid <= 0:
                    raise ValueError(f"Invalid call in {sequence!r}")
                stacks[seat] -= paid
                committed[seat] += paid
                record(seat, "call", paid)
            elif action == "k":
                if current_bet != committed[seat]:
                    raise ValueError(f"Invalid check in {sequence!r}")
                record(seat, "check", 0)
            else:
                folded = seat
                record(seat, "fold", 0)
            seat = 1 - seat
    return names, stacks, committed, seat, street_index, increment, folded, log


def encode_action(reply, owed, current_bet, minimum, maximum, opponent_stack):
    action = reply["action"]
    if action == "fold":
        return "f" if owed else "k"
    if action in ("call", "check"):
        return "c" if owed else "k"
    if action not in ("raise", "all_in"):
        raise ValueError(f"Unknown bot action {action!r}")
    if maximum <= current_bet or not opponent_stack:
        return "c" if owed else "k"
    target = maximum if action == "all_in" else int(reply["amount"])
    return f"b{min(max(target, minimum), maximum)}"


def play_hand(session, bot, hand_num, history):
    response = session.request("new_hand")
    for _ in range(100):
        hero = response["client_pos"]
        names, stacks, committed, seat, street, increment, folded, log = parse_history(
            response["action"], hero, hand_num
        )
        if "winnings" in response:
            winnings = response["winnings"]
            own_paid, opponent_paid = STACK - stacks[hero], STACK - stacks[1 - hero]
            matched = min(own_paid, opponent_paid)
            possible = (
                {-own_paid} if folded == hero else {opponent_paid} if folded == 1 - hero
                else {-matched, 0, matched}
            )
            if winnings not in possible:
                raise RuntimeError(f"Slumbot settlement {winnings} disagrees with the betting history.")
            history.extend(log)
            if folded is None and response.get("bot_hole_cards"):
                for player, cards in ((hero, response["hole_cards"]), (1 - hero, response["bot_hole_cards"])):
                    history.append(dict(hand_num=hand_num, seat=player, bot_id=names[player],
                                        action="show", amount=0, street="showdown", cards=cards))
            return winnings, hero

        if seat != hero or folded is not None or len(response["board"]) != (0, 3, 4, 5)[street]:
            raise RuntimeError("Slumbot response is inconsistent with the betting history.")
        current_bet = max(committed)
        owed = current_bet - committed[hero]
        minimum = current_bet + increment
        players = [
            dict(seat=p, bot_id=names[p], stack=stacks[p], bet_this_street=committed[p],
                 state="active" if stacks[p] else "all_in", is_folded=False,
                 is_all_in=not stacks[p], hole_cards=None)
            for p in range(2)
        ]
        state = dict(
            type="action_request", hand_num=hand_num, hand_id=f"slumbot-{hand_num}",
            street=STREETS[street], your_bot_id=HERO, seat_to_act=hero,
            your_cards=response["hole_cards"], community_cards=response["board"],
            pot=2 * STACK - sum(stacks), amount_owed=owed, can_check=owed == 0,
            current_bet=current_bet, min_raise_to=minimum, your_stack=stacks[hero],
            your_bet_this_street=committed[hero], players=players, action_log=log,
            match_action_log=(list(history) + log)[-200:], time_limit_ms=2000,
            button_seat=1, sb_seat=1, bb_seat=0,
            blinds=dict(sb=BIG_BLIND // 2, bb=BIG_BLIND, ante=0),
        )
        reply = bot.act(state, 2000)
        if reply.get("error"):
            raise RuntimeError(f"Tournament bot decision failed: {reply['error']}")
        action = encode_action(reply, owed, current_bet, minimum,
                               committed[hero] + stacks[hero], stacks[1 - hero])
        response = session.request("act", incr=action)
    raise RuntimeError("Slumbot hand did not finish within the decision limit.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hands", type=int, default=12)
    parser.add_argument("--bot", type=Path, default=ROOT / "my_bots/72_offsuit")
    args = parser.parse_args()
    if args.hands < 1 or not (args.bot / "bot.py").is_file():
        parser.error("Use a positive hand count and a bot directory containing bot.py.")

    session = SlumbotSession()
    bot = BotProcess(HERO, str(args.bot.resolve()), seed=7, use_docker=True)
    net = completed = 0
    seats = Counter()
    history = deque(maxlen=200)
    try:
        if not bot.warmup().get("ok"):
            raise RuntimeError(f"Tournament sandbox warmup failed: {bot.errors}")
        for hand in range(args.hands):
            winnings, hero = play_hand(session, bot, hand, history)
            net += winnings
            completed += 1
            seats[hero] += 1
            print(f"Hand {completed} ({'SB' if hero else 'BB'}): {winnings:+,} chips", flush=True)
    except (OSError, URLError, ValueError, KeyError, RuntimeError) as error:
        print(f"Benchmark stopped after {completed} completed hands: {error}", file=sys.stderr)
    finally:
        bot.stop()
    if completed:
        print(f"{completed} hands ({seats[1]} SB, {seats[0]} BB), {net:+,} chips, "
              f"{net / completed:+,.1f} bb/100")
    return int(completed != args.hands)


if __name__ == "__main__":
    raise SystemExit(main())
