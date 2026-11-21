"""Heads-up Texas Hold'em bot for 72 Offsuit.

See the README's "What your bot sees" section for the game-state fields and
action format. Run `make validate BOT=my_bots/72_offsuit` before submitting.
"""

import random
from itertools import combinations

import eval7
import numpy as np

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
      return []

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
    return log[overlap:]

  def action_probabilities(self, street, facing_bet):
    """Return posterior means for a street and opponent betting context."""
    alpha = self._alpha[street, facing_bet]
    total = sum(alpha.values())
    return {action: count / total for action, count in alpha.items()}


_opponent = Opponent()


_CARD_NAMES = tuple(str(card) for card in _DECK)
_CARD_IDS = {name: index for index, name in enumerate(_CARD_NAMES)}
_HAND_COMBOS = np.asarray(
  list(combinations(range(len(_DECK)), 2)), dtype=np.uint8
)
_COMBO_COUNT = len(_HAND_COMBOS)
_RANK_VALUES = {rank: value for value, rank in enumerate("23456789TJQKA", 2)}


def _card_ids(cards):
  return {_CARD_IDS[card] for card in cards}


def _preflop_strengths():
  """Return a rough 0..1 prior strength for each exact preflop combo."""
  strengths = np.empty(_COMBO_COUNT, dtype=np.float32)
  for index, (first, second) in enumerate(_HAND_COMBOS):
    first_rank = _RANK_VALUES[_CARD_NAMES[first][0]]
    second_rank = _RANK_VALUES[_CARD_NAMES[second][0]]
    high, low = max(first_rank, second_rank), min(first_rank, second_rank)
    suited = _CARD_NAMES[first][1] == _CARD_NAMES[second][1]
    pair_bonus = 0.35 if first_rank == second_rank else 0.0
    suited_bonus = 0.06 if suited else 0.0
    connected_bonus = max(0, 5 - high + low) * 0.008
    strengths[index] = min(
      1.0,
      (high - 2) / 12.0 * 0.62
      + (low - 2) / 12.0 * 0.22
      + pair_bonus + suited_bonus + connected_bonus,
    )
  return strengths


_PREFLOP_STRENGTHS = _preflop_strengths()


class OpponentRange:
  """A log-weighted distribution over all exact opponent hole-card combos."""

  _ACTION_SHARPNESS = {
    "fold": 1.0,
    "check": 0.8,
    "call": 1.1,
    "raise": 1.5,
    "all_in": 1.8,
  }

  def __init__(self):
    self._log_weights = np.zeros(_COMBO_COUNT, dtype=np.float32)
    self._hand_num = None

  def reset(self, hand_num=None):
    self._log_weights.fill(0.0)
    self._hand_num = hand_num

  def _strengths(self, game_state):
    board = game_state.get("community_cards", [])
    if not board:
      return _PREFLOP_STRENGTHS

    board_cards = [eval7.Card(card) for card in board]
    scores = np.full(_COMBO_COUNT, -np.inf, dtype=np.float32)
    for index, (first, second) in enumerate(_HAND_COMBOS):
      if _CARD_NAMES[first] in board or _CARD_NAMES[second] in board:
        continue
      scores[index] = eval7.evaluate(
        [eval7.Card(_CARD_NAMES[first]), eval7.Card(_CARD_NAMES[second])]
        + board_cards
      )

    legal = np.isfinite(scores)
    legal_scores = scores[legal]
    order = np.argsort(legal_scores)
    ranks = np.empty(len(legal_scores), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(legal_scores), dtype=np.float32)
    strengths = np.zeros(_COMBO_COUNT, dtype=np.float32)
    strengths[legal] = ranks
    return strengths

  def _legal_mask(self, game_state):
    known = _card_ids(
      list(game_state.get("your_cards", []))
      + list(game_state.get("community_cards", []))
    )
    return np.asarray(
      [first not in known and second not in known for first, second in _HAND_COMBOS],
      dtype=bool,
    )

  @staticmethod
  def _likelihood(action, strength):
    """Soft action likelihood; never hard-eliminates a merely unlikely hand."""
    strength = np.clip(strength, 0.0, 1.0)
    sharpness = OpponentRange._ACTION_SHARPNESS.get(action, 1.0)
    if action == "fold":
      likelihood = 1.0 - 0.90 * strength
    elif action == "check":
      likelihood = 1.0 - 0.55 * strength
    elif action == "call":
      likelihood = 0.15 + 0.85 * (1.0 - np.abs(strength - 0.58))
    else:
      likelihood = 0.05 + 0.95 * strength
    return np.maximum(0.01, likelihood) ** sharpness

  def observe(self, game_state, events):
    """Condition the range on newly observed opponent actions."""
    hand_num = game_state.get("hand_num")
    if hand_num != self._hand_num:
      self.reset(hand_num)

    legal = self._legal_mask(game_state)
    self._log_weights[~legal] = -np.inf
    strengths = self._strengths(game_state)
    opponent_id = game_state.get("your_bot_id")

    for event in events:
      if event.get("bot_id") == opponent_id:
        continue
      action = event.get("action")
      if action == "show" and len(event.get("cards", [])) == 2:
        shown = set(event["cards"])
        exact = np.asarray(
          [{_CARD_NAMES[first], _CARD_NAMES[second]} == shown
           for first, second in _HAND_COMBOS],
          dtype=bool,
        )
        self._log_weights[~exact] = -np.inf
        continue
      if action not in self._ACTION_SHARPNESS:
        continue
      likelihood = self._likelihood(action, strengths)
      self._log_weights[legal] += np.log(likelihood[legal])

    self._log_weights[~legal] = -np.inf

  def probabilities(self, game_state):
    legal = self._legal_mask(game_state)
    values = self._log_weights.copy()
    values[~legal] = -np.inf
    maximum = np.max(values)
    if not np.isfinite(maximum):
      values[legal] = 0.0
      maximum = 0.0
    probabilities = np.exp(values - maximum)
    probabilities[~legal] = 0.0
    total = probabilities.sum()
    if total <= 0.0:
      probabilities[legal] = 1.0 / max(1, legal.sum())
      return probabilities
    return probabilities / total

  def sample(self, game_state, count, rng):
    probabilities = self.probabilities(game_state)
    return rng.choice(_COMBO_COUNT, size=count, p=probabilities)


_opponent_range = OpponentRange()
_RANGE_RNG = np.random.default_rng()


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


def weighted_monte_carlo(game_state, opponent_range=None, trials=512):
  """Estimate equity by sampling opponent hands from the live weighted range."""
  if opponent_range is None:
    opponent_range = _opponent_range

  hole_cards = [eval7.Card(card) for card in game_state["your_cards"]]
  board = [eval7.Card(card) for card in game_state["community_cards"]]
  known_ids = _card_ids(game_state["your_cards"] + game_state["community_cards"])
  cards_to_come = 5 - len(board)
  sampled_combos = opponent_range.sample(game_state, trials, _RANGE_RNG)

  wins = 0
  for combo_index in sampled_combos:
    first, second = _HAND_COMBOS[combo_index]
    opponent_cards = [eval7.Card(_CARD_NAMES[first]), eval7.Card(_CARD_NAMES[second])]
    blocked = known_ids | {int(first), int(second)}
    available = [card_id for card_id in range(len(_DECK)) if card_id not in blocked]
    if cards_to_come:
      draw_ids = _RANGE_RNG.choice(available, size=cards_to_come, replace=False)
      runout = [eval7.Card(_CARD_NAMES[int(card_id)]) for card_id in draw_ids]
    else:
      runout = []
    final_board = board + runout
    if eval7.evaluate(hole_cards + final_board) > eval7.evaluate(opponent_cards + final_board):
      wins += 1
  return wins / max(1, trials)


def _equity(game_state: dict, equity=None) -> float:
    """Use supplied equity, or estimate it from the weighted opponent range."""
    if equity is None:
        equity = weighted_monte_carlo(game_state)
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
    _opponent_range.reset()
    return {"action": "check"}

  new_events = _opponent.observe(game_state)
  _opponent_range.observe(game_state, new_events)
  equity = weighted_monte_carlo(game_state, _opponent_range)

  first_card, second_card = game_state["your_cards"]
  pot = game_state["pot"]

  if first_card[0] == second_card[0] and first_card[0] in "AKQJ":
    raise_to = max(game_state["min_raise_to"], game_state["current_bet"] + pot)
    fold_equity = _opponent.action_probabilities(
      game_state["street"], True
    )["fold"]
    return {
      "action": "raise",
      "amount": raise_to,
      "debug": {
        "equity": equity,
        "ev": ev_calc_raise(game_state, raise_to, equity, fold_equity),
        "fold_equity": fold_equity,
      },
    }
  if game_state["can_check"]:
    return {
      "action": "check",
      "debug": {"equity": equity, "ev": ev_calc_check(game_state, equity)},
    }

  call_ev = ev_calc_call(game_state, equity)
  if call_ev >= 0.0:
    return {"action": "call", "debug": {"equity": equity, "ev": call_ev}}
  return {"action": "fold", "debug": {"equity": equity, "ev": ev_calc_fold(game_state)}}
