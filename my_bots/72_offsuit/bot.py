"""Heads-up Texas Hold'em bot for 72 Offsuit.

See the README's "What your bot sees" section for the game-state fields and
action format. Run `make validate BOT=my_bots/72_offsuit` before submitting.
"""

import random
from itertools import combinations

import eval7

_DECK = tuple(eval7.Deck().cards)
_MONTE_CARLO_TRIALS = 5000


class Opponent:
  """Dirichlet posteriors by street and whether the opponent owes chips."""

  _STREETS = ("preflop", "flop", "turn", "river")
  _ACTIONS = {
    False: ("fold", "check", "raise", "all_in"),
    True: ("fold", "call", "raise", "all_in"),
  }

  def __init__(self):
    self._alpha = {
      (street, facing_bet): dict.fromkeys(actions, 1)
      for street in self._STREETS
      for facing_bet, actions in self._ACTIONS.items()
    }
    self._previous_log = []
    self._round = None
    self._bets = {}
    self._round_complete = False

  def observe(self, game_state):
    """Update the model once per new action in the rolling match log."""
    log = game_state.get("match_action_log")
    if not log:
      return

    # An older prefix can fall out of the rolling log between decisions.
    # Match its overlap before filtering out our actions, posts, and shows.
    overlap = min(len(log), len(self._previous_log))
    while overlap and self._previous_log[-overlap:] != log[:overlap]:
      overlap -= 1

    if not overlap:
      # If the log opens mid-round, the missing bets can change the context.
      first = log[0]
      self._round = (first["hand_num"], first["street"])
      self._bets = {}
      self._round_complete = first["action"] in ("ante", "small_blind")

    our_id = game_state["your_bot_id"]
    for event in log[overlap:]:
      street = event["street"]
      if street not in self._STREETS:
        continue
      betting_round = (event["hand_num"], street)
      if betting_round != self._round:
        self._round = betting_round
        self._bets = {}
        self._round_complete = True

      actor = event["bot_id"]
      action = event["action"]
      committed = self._bets.get(actor, 0)
      if actor != our_id and self._round_complete:
        facing_bet = committed < max(self._bets.values(), default=0)
        alpha = self._alpha[street, facing_bet]
        if action in alpha:
          alpha[action] += 1

      if action in ("small_blind", "big_blind", "call"):
        self._bets[actor] = committed + event["amount"]
      elif action in ("raise", "all_in"):
        self._bets[actor] = event["amount"]
    self._previous_log = [event.copy() for event in log]

  def action_probabilities(self, street, facing_bet):
    """Return posterior means for a street and opponent betting context."""
    alpha = self._alpha[street, facing_bet]
    total = sum(alpha.values())
    return {action: count / total for action, count in alpha.items()}


_opponent = Opponent()


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
  """Pick an action using the starter strategy.

  The engine sends a warmup request at the start of each match. During play,
  a reply can include a JSON-serializable "debug" value alongside the action.
  """
  global _opponent
  if game_state.get("type") == "warmup":
    _opponent = Opponent()
    return {"action": "check"}

  _opponent.observe(game_state)

  first_card, second_card = game_state["your_cards"]
  pot = game_state["pot"]

  if first_card[0] == second_card[0] and first_card[0] in "AKQJ":
    raise_to = max(game_state["min_raise_to"], game_state["current_bet"] + pot)
    return {"action": "raise", "amount": raise_to}
  if game_state["can_check"]:
    return {"action": "check"}
  action = "call" if game_state["amount_owed"] <= pot * 0.25 else "fold"
  return {"action": action}
