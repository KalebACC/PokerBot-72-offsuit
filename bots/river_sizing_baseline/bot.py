"""Heads-up Texas Hold'em bot for 72 Offsuit.

See the README's "What your bot sees" section for the game-state fields and
action format. Run `make validate BOT=my_bots/72_offsuit` before submitting.
"""

import random
from itertools import combinations

import eval7

_DECK = tuple(eval7.Deck().cards)
_MONTE_CARLO_TRIALS = 5000


def _preflop_strength(first, second):
  """Give premium and strong starting hands more mass in a value range."""
  ranks = sorted((first[0], second[0]), key="23456789TJQKA".index, reverse=True)
  hand = "".join(ranks)
  suited = first[1] == second[1]
  if hand in ("AA", "KK", "QQ", "JJ", "AK") or (hand == "AQ" and suited):
    return 10
  if hand in ("TT", "99", "88", "77", "AQ") or (suited and hand in ("AJ", "AT", "KQ")):
    return 4
  return 1


_PREFLOP_STRENGTH = {
  (first, second): _preflop_strength(str(first), str(second))
  for first, second in combinations(_DECK, 2)
}
_MEAN_PREFLOP_STRENGTH = sum(_PREFLOP_STRENGTH.values()) / len(_PREFLOP_STRENGTH)


class Opponent:
  """Dirichlet posteriors by street and whether the opponent owes chips."""

  _STREETS = ("preflop", "flop", "turn", "river")
  _ACTIONS = {
    False: ("fold", "check", "raise", "all_in"),
    True: ("fold", "call", "raise", "all_in"),
  }
  _LARGE_BET_PRICE = 0.4
  _RIVER_SIZE_PRIOR = 8

  def __init__(self):
    self._alpha = {
      (street, facing_bet): dict.fromkeys(actions, 1)
      for street in self._STREETS
      for facing_bet, actions in self._ACTIONS.items()
    }
    self._previous_log = []
    self._round = None
    self._bets = {}
    self._pot = None
    self._river_responses = {
      large_bet: dict.fromkeys(self._ACTIONS[True], 0) for large_bet in (False, True)
    }
    self._round_complete = False
    self._preflop_raiser = None
    self._reraise_chances = 0
    self._reraises = 0

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
      self._pot = 0 if self._round_complete else None
      self._preflop_raiser = None

    our_id = game_state["your_bot_id"]
    for event in log[overlap:]:
      street = event["street"]
      if street not in self._STREETS:
        continue
      betting_round = (event["hand_num"], street)
      if betting_round != self._round:
        if event["hand_num"] != self._round[0]:
          self._pot = 0
        self._round = betting_round
        self._bets = {}
        self._round_complete = True
        self._preflop_raiser = None

      actor = event["bot_id"]
      action = event["action"]
      committed = self._bets.get(actor, 0)
      highest_bet = max(self._bets.values(), default=0)
      if actor != our_id and self._round_complete:
        facing_bet = committed < highest_bet
        alpha = self._alpha[street, facing_bet]
        if action in alpha:
          alpha[action] += 1
          if street == "river" and facing_bet and self._pot:
            large_bet = (highest_bet - committed) / self._pot >= self._LARGE_BET_PRICE
            self._river_responses[large_bet][action] += 1
          if street == "preflop" and facing_bet and self._preflop_raiser == our_id:
            self._reraise_chances += 1
            self._reraises += action in ("raise", "all_in") and event["amount"] > highest_bet

      if action in ("small_blind", "big_blind", "call"):
        self._bets[actor] = committed + event["amount"]
      elif action in ("raise", "all_in"):
        self._bets[actor] = event["amount"]
        if street == "preflop" and event["amount"] > highest_bet:
          self._preflop_raiser = actor if action == "raise" else None
      if self._pot is not None:
        self._pot += event["amount"] if action == "ante" else self._bets.get(actor, 0) - committed
    self._previous_log = [event.copy() for event in log]

  def action_probabilities(self, street, facing_bet):
    """Return posterior means for a street and opponent betting context."""
    alpha = self._alpha[street, facing_bet]
    total = sum(alpha.values())
    return {action: count / total for action, count in alpha.items()}

  def river_response_probabilities(self, amount_owed, pot, street_probabilities):
    """Blend observed responses at this price with the overall river model."""
    large_bet = amount_owed / pot >= self._LARGE_BET_PRICE
    counts = self._river_responses[large_bet]
    total = sum(counts.values()) + self._RIVER_SIZE_PRIOR
    return {
      action: (count + self._RIVER_SIZE_PRIOR * street_probabilities[action]) / total
      for action, count in counts.items()
    }

  def preflop_range_tilt(self, game_state):
    """Mix in a value range only when the opponent reraises our preflop bet."""
    if game_state.get("street") != "preflop" or not game_state.get("amount_owed"):
      return 0.0
    our_id = game_state["your_bot_id"]
    our_raise = False
    for event in game_state.get("action_log", []):
      if event["action"] not in ("raise", "all_in"):
        continue
      if event["bot_id"] == our_id:
        our_raise = True
      elif our_raise and event["amount"] > game_state["your_bet_this_street"]:
        # Start with three reraises in ten hypothetical opportunities.
        rate = (self._reraises + 3) / (self._reraise_chances + 10)
        return 0.6 * max(0.0, 1.0 - rate / 0.65)
    return 0.0

  def river_range_weights(self, game_state, hand_scores):
    """Weight possible hands after an observed river bet or raise."""
    amount_owed = game_state.get("amount_owed", 0)
    if not amount_owed:
      return None

    our_id = game_state.get("your_bot_id")
    last_response = next(
      (event["action"] for event in reversed(game_state.get("action_log", []))
       if event["street"] == "river" and event["bot_id"] != our_id),
      None,
    )
    if last_response not in ("raise", "all_in"):
      return None

    # Larger bets tilt the range toward higher ranks. The floor retains bluffs.
    pot_before_bet = max(1, game_state["pot"] - amount_owed)
    pressure = min(1.0, amount_owed / pot_before_bet)
    exponent = 1.0 + 3.0 * pressure
    counts = {}
    for score in hand_scores:
      counts[score] = counts.get(score, 0) + 1

    weights = {}
    weaker = 0
    for score in sorted(counts):
      count = counts[score]
      percentile = (weaker + 0.5 * count) / len(hand_scores)
      weights[score] = 0.15 + 0.85 * percentile ** exponent
      weaker += count
    return weights


_opponent = Opponent()


def monte_carlo(game_state):
  """Return the probability of winning outright at a heads-up showdown."""
  wins, _ = _showdown_probabilities(game_state)
  return wins


def _showdown_probabilities(game_state, opponent=None):
  """Return win and tie probabilities, optionally weighting river hands."""
  hole_cards = [eval7.Card(card) for card in game_state["your_cards"]]
  board = [eval7.Card(card) for card in game_state["community_cards"]]
  known_cards = set(hole_cards + board)
  unseen = [card for card in _DECK if card not in known_cards]
  cards_to_come = 5 - len(board)

  wins = 0
  ties = 0
  if cards_to_come == 0:
    our_score = eval7.evaluate(hole_cards + board)
    scores = [eval7.evaluate(board + [first, second]) for first, second in combinations(unseen, 2)]
    weights = opponent.river_range_weights(game_state, scores) if opponent else None
    total_weight = 0.0
    for score in scores:
      weight = weights[score] if weights is not None else 1.0
      total_weight += weight
      if our_score > score:
        wins += weight
      elif our_score == score:
        ties += weight
    return wins / total_weight, ties / total_weight

  tilt = opponent.preflop_range_tilt(game_state) if opponent else 0.0
  total_weight = 0.0
  for _ in range(_MONTE_CARLO_TRIALS):
    draw = random.sample(unseen, 2 + cards_to_come)
    final_board = board + draw[2:]
    our_score = eval7.evaluate(hole_cards + final_board)
    their_score = eval7.evaluate(draw[:2] + final_board)
    weight = 1.0
    if tilt:
      first, second = draw[:2]
      strength = _PREFLOP_STRENGTH.get((first, second)) or _PREFLOP_STRENGTH[second, first]
      weight += tilt * (strength / _MEAN_PREFLOP_STRENGTH - 1.0)
    total_weight += weight
    if our_score > their_score:
      wins += weight
    elif our_score == their_score:
      ties += weight
  return wins / total_weight, ties / total_weight


def _equity(game_state: dict, equity=None) -> float:
  """Use supplied showdown equity, or estimate it with Monte Carlo."""
  if equity is None:
    wins, ties = _showdown_probabilities(game_state)
    equity = wins + 0.5 * ties
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
    *,
    response_probabilities=None,
    reraise_equity=None,
) -> float:
  """EV of raising to a street total, with optional opponent responses."""
  if response_probabilities is None:
    return ev_calc_bet(game_state, raise_to, equity, fold_equity, opponent_call_chips)

  equity = _equity(game_state, equity)
  if reraise_equity is None:
    reraise_equity = equity
  pot = game_state["pot"]
  current_bet = game_state["current_bet"]
  raise_to = _effective_total_bet(game_state, raise_to)
  chips_added = _additional_chips(game_state, raise_to)
  matched_pot = pot + chips_added + _opponent_call_chips(game_state, raise_to)
  call_ev = equity * matched_pot - chips_added

  effective_max = _effective_total_bet(
    game_state, game_state["your_bet_this_street"] + game_state["your_stack"]
  )
  # Model a regular reraise as pot-sized, subject to the minimum and stacks.
  min_increment = max(game_state["min_raise_to"] - current_bet, raise_to - current_bet)
  reraise_to = min(effective_max, raise_to + max(min_increment, matched_pot))

  def response_ev(total):
    extra_to_call = total - raise_to
    if extra_to_call <= 0:
      return call_ev
    showdown_ev = reraise_equity * (matched_pot + 2 * extra_to_call)
    return max(-chips_added, showdown_ev - chips_added - extra_to_call)

  return (
    response_probabilities["fold"] * pot
    + response_probabilities["call"] * call_ev
    + response_probabilities["raise"] * response_ev(reraise_to)
    + response_probabilities["all_in"] * response_ev(effective_max)
  )


def decide(game_state: dict) -> dict:
  """Choose the candidate with the highest approximate chip EV."""
  global _opponent
  if game_state.get("type") == "warmup":
    _opponent = Opponent()
    return {"action": "check"}

  _opponent.observe(game_state)
  wins, ties = _showdown_probabilities(game_state, _opponent)
  equity = wins + 0.5 * ties
  can_check = game_state["can_check"]
  best_action = {"action": "check" if can_check else "fold"}
  best_ev = ev_calc_check(game_state, equity) if can_check else ev_calc_fold(game_state)
  if not can_check:
    call_ev = ev_calc_call(game_state, equity)
    if call_ev > best_ev:
      best_action, best_ev = {"action": "call"}, call_ev

  fold_probability = None
  min_raise = game_state["min_raise_to"]
  # Avoid bloating the pot with a bluff after a reraise from a tighter range.
  preflop_tilt = _opponent.preflop_range_tilt(game_state)
  avoid_reraise = preflop_tilt >= 0.15 and equity < 0.5
  if (not avoid_reraise and _opponent_stack(game_state)
      and _effective_total_bet(game_state, min_raise) == min_raise):
    responses = _opponent.action_probabilities(
      game_state["street"], facing_bet=True
    )
    fold_probability = responses["fold"]
    pot_after_call = game_state["pot"] + game_state["amount_owed"]
    current_bet = game_state["current_bet"]
    facing_preflop_raise = (
      game_state["street"] == "preflop"
      and not can_check
      and any(
        event["bot_id"] != game_state["your_bot_id"] and event["action"] in ("raise", "all_in")
        for event in game_state.get("action_log", [])
      )
    )
    candidate_sizes = [min_raise, current_bet + pot_after_call // 2]
    if not facing_preflop_raise:
      candidate_sizes.append(current_bet + pot_after_call)
    targets = {
      _effective_total_bet(game_state, max(min_raise, target))
      for target in candidate_sizes
    }
    for target in sorted(targets):
      candidate_responses = responses
      if game_state["street"] == "river":
        candidate_responses = _opponent.river_response_probabilities(
          _opponent_call_chips(game_state, target),
          game_state["pot"] + _additional_chips(game_state, target),
          responses,
        )
      raise_ev = ev_calc_raise(game_state, target, equity, response_probabilities=candidate_responses)
      if raise_ev > best_ev:
        best_action, best_ev = {"action": "raise", "amount": target}, raise_ev
        fold_probability = candidate_responses["fold"]

  best_action["debug"] = {"equity": equity, "fold_probability": fold_probability, "ev": best_ev}
  return best_action
