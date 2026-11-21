"""
Bot template. Copy this folder to my_bots/<your_team>/ and write decide().

  - decide() is called once per decision and has 2 seconds to return, or your
    bot folds. Crashes and invalid actions also fold; your bot stays in the game.
  - You may use the Python 3.10 standard library, eval7, numpy, scipy, treys
    and scikit-learn. Nothing else is installed in the sandbox.
  - No network, no file writes, no subprocesses, no eval/exec. You may read
    your own data/ folder, at import time only.
  - Check your bot before submitting:  make validate BOT=my_bots/<your_team>

The full list of what decide() receives is in the README ("What your bot
sees"); the most useful keys are summarised below.

RETURN FORMAT:
  {"action": "fold"}
  {"action": "check"}                       # only when can_check is True
  {"action": "call"}
  {"action": "raise", "amount": 1200}       # amount = your TOTAL bet this street
  {"action": "all_in"}

CARDS are strings like "As" (ace of spades) or "Td" (ten of diamonds).
"""

import random
from itertools import combinations

import eval7

_DECK = tuple(eval7.Deck().cards)
_MONTE_CARLO_TRIALS = 5000


def monte_carlo(game_state):
    """Return the probability of winning outright at a heads-up showdown.

    Assume uniformly random opponent cards and board runouts from the unseen
    deck. Ties do not count as wins. Enumerate all opponent hands on the river.
    """
    hole_cards = [eval7.Card(card) for card in game_state["your_cards"]]
    board = [eval7.Card(card) for card in game_state["community_cards"]]
    known_cards = set(hole_cards + board)
    unseen = [card for card in _DECK if card not in known_cards]
    cards_to_come = 5 - len(board)

    wins = 0
    if cards_to_come == 0:
        our_score = eval7.evaluate(hole_cards + board)
        for first, second in combinations(unseen, 2):
            if our_score > eval7.evaluate(board + [first, second]):
                wins += 1
        return wins / (len(unseen) * (len(unseen) - 1) // 2)

    for _ in range(_MONTE_CARLO_TRIALS):
        draw = random.sample(unseen, 2 + cards_to_come)
        final_board = board + draw[2:]
        if eval7.evaluate(hole_cards + final_board) > eval7.evaluate(draw[:2] + final_board):
            wins += 1
    return wins / _MONTE_CARLO_TRIALS


def _equity(game_state: dict, equity=None) -> float:
    """Use supplied showdown equity, or estimate it with Monte Carlo."""
    if equity is None:
        equity = monte_carlo(game_state)
    return max(0.0, min(1.0, float(equity)))


def _fold_equity(fold_equity: float) -> float:
    """Keep fold-equity inputs in the probability range."""
    return max(0.0, min(1.0, float(fold_equity)))


def _additional_chips(game_state: dict, total_bet: int) -> int:
    """Return chips added by a bet/raise; amounts are totals this street."""
    street_bet = int(game_state.get("your_bet_this_street", 0))
    stack = int(game_state.get("your_stack", total_bet))
    return max(0, min(int(total_bet) - street_bet, stack))


def _opponent_stack(game_state: dict) -> int | None:
    """Return the active heads-up opponent's chips behind, if known."""
    for player in game_state.get("players", []):
        if (
            player.get("bot_id") != game_state.get("your_bot_id")
            and player.get("state", "active") == "active"
        ):
            stack = player.get("stack")
            return None if stack is None else int(stack)

    return None


def _effective_total_bet(game_state: dict, total_bet: int) -> int:
    """Cap a target before calculating either player's contribution."""
    own_max = int(game_state.get("your_bet_this_street", 0)) + int(
        game_state.get("your_stack", total_bet)
    )
    opponent_stack = _opponent_stack(game_state)
    if opponent_stack is None:
        return min(int(total_bet), own_max)
    opponent_max = int(game_state.get("current_bet", 0)) + opponent_stack
    return min(int(total_bet), own_max, opponent_max)


def _opponent_call_chips(game_state: dict, total_bet: int) -> int:
    """Estimate one opponent's chips added when calling our total bet."""
    call_amount = max(0, int(total_bet) - int(game_state.get("current_bet", 0)))
    opponent_stack = _opponent_stack(game_state)
    if opponent_stack is None:
        return call_amount
    return min(call_amount, opponent_stack)


def ev_calc_fold(game_state: dict) -> float:
    """EV of folding now, measured from the current decision point."""
    return 0.0


def ev_calc_check(game_state: dict, equity=None) -> float:
    """EV of checking: showdown equity multiplied by the current pot."""
    return _equity(game_state, equity) * float(game_state["pot"])


def ev_calc_call(game_state: dict, equity=None) -> float:
    """EV of calling heads-up, assuming no further betting."""
    equity = _equity(game_state, equity)
    amount_owed = float(game_state["amount_owed"])
    your_stack = float(game_state.get("your_stack", amount_owed))

    call_cost = min(amount_owed, your_stack)
    uncalled_chips = amount_owed - call_cost
    showdown_pot = float(game_state["pot"]) - uncalled_chips + call_cost

    return equity * showdown_pot - call_cost

def ev_calc_bet(
    game_state: dict,
    bet_to: int,
    equity=None,
    fold_equity: float = 0.0,
    opponent_call_chips=None,
) -> float:
    """EV of betting heads-up, assuming a call leads to showdown."""
    equity = _equity(game_state, equity)
    fold_equity = _fold_equity(fold_equity)

    opponent_street_bet = int(game_state.get("current_bet", 0))
    effective_bet_to = _effective_total_bet(game_state, bet_to)

    if opponent_call_chips is not None:
        max_call = max(0, int(opponent_call_chips))
        effective_bet_to = min(
            effective_bet_to, opponent_street_bet + max_call
        )

    chips_added = _additional_chips(game_state, effective_bet_to)
    call_cost = max(0, effective_bet_to - opponent_street_bet)

    pot = float(game_state["pot"])
    showdown_pot = pot + chips_added + call_cost
    showdown_ev = equity * showdown_pot - chips_added

    return fold_equity * pot + (1.0 - fold_equity) * showdown_ev

def ev_calc_raise(
    game_state: dict,
    raise_to: int,
    equity=None,
    fold_equity: float = 0.0,
    opponent_call_chips=None,
) -> float:
    """EV of raising to ``raise_to``; this is not a raise-by amount."""
    return ev_calc_bet(
        game_state,
        raise_to,
        equity,
        fold_equity,
        opponent_call_chips,
    )


def decide(game_state: dict) -> dict:
    """
    game_state (the most useful keys):
      your_cards        ["Ah", "Kh"]
      community_cards   [] preflop, then 3, 4, 5 cards
      street            "preflop" | "flop" | "turn" | "river"
      pot, current_bet, min_raise_to, amount_owed, can_check
      your_stack, your_bet_this_street, your_bot_id
      blinds            {"sb", "bb", "ante"} for this hand (they rise every 20 hands)
      hand_num          hand number in this match, from 0
      button_seat, sb_seat, bb_seat, seat_to_act
      players           public info per seat: seat, bot_id, stack, state, bet_this_street, ...
      action_log        every action this hand: {"seat", "bot_id", "action", "amount", "street"}
      match_action_log  the last 200 actions across the match, with "hand_num", plus
                        {"action": "show", "cards": [...]} for each player at a showdown

    Debugging: anything you print() is saved with the decision, and you can add
    a JSON-serialisable "debug" value to what you return, e.g.
    {"action": "call", "debug": {"equity": 0.4}}. Run with: python3 play.py --show
    """
    if game_state.get("type") == "warmup":     # one call before hand 1; load things here if you like
        return {"action": "check"}

    # ── Your strategy goes here. This example is deliberately simple. ──────────
    cards = game_state["your_cards"]
    pot = game_state["pot"]
    owed = game_state["amount_owed"]

    ranks = [c[0] for c in cards]
    if ranks[0] == ranks[1] and ranks[0] in "AKQJ":           # big pocket pair: raise
        raise_to = max(game_state["min_raise_to"], game_state["current_bet"] + pot)
        return {"action": "raise", "amount": raise_to}
    if game_state["can_check"]:
        return {"action": "check"}
    if owed <= pot * 0.25:                                     # cheap to continue: call
        return {"action": "call"}
    return {"action": "fold"}
