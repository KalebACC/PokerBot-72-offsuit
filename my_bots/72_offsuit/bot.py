"""Heads-up Texas Hold'em bot for 72 Offsuit.

See the README's "What your bot sees" section for the game-state fields and
action format. Run `make validate BOT=my_bots/72_offsuit` before submitting.
"""

import random
from itertools import combinations

import eval7

_DECK = tuple(eval7.Deck().cards)
_MONTE_CARLO_TRIALS = 20_000


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
  (first, second): _preflop_strength(str(first), str(second)) ** 2
  for first, second in combinations(_DECK, 2)
}
_MEAN_PREFLOP_STRENGTH = sum(_PREFLOP_STRENGTH.values()) / len(_PREFLOP_STRENGTH)


def _hand_percentiles(scores):
  """Rank hand scores by percentile, giving ties the same rank."""
  counts = {}
  for score in scores:
    counts[score] = counts.get(score, 0) + 1
  weaker = 0
  for score in sorted(counts):
    count = counts[score]
    yield score, (weaker + 0.5 * count) / len(scores)
    weaker += count


class Opponent:
  """Dirichlet posteriors by street and whether the opponent owes chips."""

  _STREETS = ("preflop", "flop", "turn", "river")
  _ACTIONS = {
    False: ("fold", "check", "raise", "all_in"),
    True: ("fold", "call", "raise", "all_in"),
  }
  _LARGE_BET_PRICE = 0.4
  _RIVER_SIZE_PRIOR = 8

  @classmethod
  def is_large_bet(cls, amount_owed, pot):
    return amount_owed / pot >= cls._LARGE_BET_PRICE

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
    self._preflop_has_raise = False
    self._preflop_raiser = None
    self._last_reraise_sample_hand = None
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
      self._preflop_has_raise = False
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
        self._preflop_has_raise = False
        self._preflop_raiser = None

      actor = event["bot_id"]
      action = event["action"]
      committed = self._bets.get(actor, 0)
      highest_bet = max(self._bets.values(), default=0)
      if actor != our_id and self._round_complete:
        facing_bet = committed < highest_bet
        alpha = self._alpha[street, facing_bet]
        # The small blind acts first even when we have made no raise.
        facing_only_blinds = street == "preflop" and facing_bet and not self._preflop_has_raise
        # An all-in total at or below the live bet is a short or exact call.
        response = "call" if action == "all_in" and facing_bet and event["amount"] <= highest_bet else action
        if response in alpha and not facing_only_blinds:
          alpha[response] += 1
          if street == "river" and facing_bet and self._pot:
            large_bet = self.is_large_bet(highest_bet - committed, self._pot)
            self._river_responses[large_bet][response] += 1
          if (street == "preflop" and facing_bet and self._preflop_raiser == our_id
              and event["hand_num"] != self._last_reraise_sample_hand):
            # Multiple raises in a single hand are correlated evidence.
            self._last_reraise_sample_hand = event["hand_num"]
            self._reraise_chances += 1
            self._reraises += action in ("raise", "all_in") and event["amount"] > highest_bet

      if action in ("small_blind", "big_blind", "call"):
        self._bets[actor] = committed + event["amount"]
      elif action in ("raise", "all_in"):
        self._bets[actor] = event["amount"]
        if street == "preflop" and event["amount"] > highest_bet:
          self._preflop_has_raise = True
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
    large_bet = self.is_large_bet(amount_owed, pot)
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
    facing_reraise = False
    opponent_raises = 0
    highest_bet = 0
    for event in game_state.get("action_log", []):
      action = event["action"]
      if action in ("small_blind", "big_blind"):
        highest_bet = max(highest_bet, event["amount"])
      if action not in ("raise", "all_in") or event["amount"] <= highest_bet:
        continue
      highest_bet = event["amount"]
      if event["bot_id"] == our_id:
        our_raise = True
        facing_reraise = False
      else:
        opponent_raises += 1
        facing_reraise = our_raise
    if not facing_reraise:
      return 0.0

    # Start with three reraises in ten hypothetical opportunities.
    rate = (self._reraises + 3) / (self._reraise_chances + 10)
    tightness = max(0.0, 1.0 - rate / 0.65)
    # A second raise on the same line represents a stronger range, especially
    # from an opponent who rarely reraises across different hands.
    depth_weight = 0.6 if opponent_raises == 1 else 1.35
    return min(0.9, depth_weight * tightness)

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
    floor = 0.15
    if game_state["your_bet_this_street"] and (pressure >= 0.4 or last_response == "all_in"):
      # A raise over our river bet filters out more marginal hands than a lead.
      exponent += 2.0 + 2.0 * pressure
      floor = 0.035 if last_response == "all_in" else 0.065
    return {
      score: floor + (1 - floor) * percentile ** exponent
      for score, percentile in _hand_percentiles(hand_scores)
    }

  def turn_response_weights(self, game_state, hand_scores):
    """Relative weights for a call and reraise at two candidate bet sizes."""
    our_id = game_state["your_bot_id"]
    aggression = sum(
      entry["street"] == "turn" and entry["bot_id"] != our_id
      and entry["action"] in ("raise", "all_in")
      for entry in game_state.get("action_log", [])
    )
    # Retain weak hands and draws, while giving more weight to made hands
    # after turn aggression and in the ranges that continue for more chips.
    call_floor = self.action_probabilities("turn", facing_bet=True)["call"] ** 2
    shapes = ((max(0.35, call_floor), 2), (0.10, 5),
              (max(0.15, call_floor), 4), (0.05, 7))
    weights = {}
    for score, percentile in _hand_percentiles(hand_scores):
      prior = 0.10 + 0.90 * percentile ** (2 + min(aggression, 3)) if aggression else 1.0
      weights[score] = tuple(
        prior * (floor + (1 - floor) * percentile ** power)
        for floor, power in shapes
      )
    return weights


_opponent = Opponent()


def monte_carlo(game_state):
  """Return the probability of winning outright at a heads-up showdown."""
  wins, _ = _showdown_probabilities(game_state)
  return wins


def _showdown_cards(game_state):
  hole_cards = [eval7.Card(card) for card in game_state["your_cards"]]
  board = [eval7.Card(card) for card in game_state["community_cards"]]
  known_cards = set(hole_cards + board)
  unseen = [card for card in _DECK if card not in known_cards]
  return hole_cards, board, unseen


def _postflop_reraise_weights(game_state, board, unseen):
  """Weight opponent hands when their aggression follows our investment."""
  owed = game_state["amount_owed"]
  if not owed:
    return None
  our_id = game_state["your_bot_id"]
  opponent_raises = [
    event for event in game_state.get("action_log", [])
    if event["street"] == game_state["street"] and event["bot_id"] != our_id
    and event["action"] in ("raise", "all_in")
  ]
  if not opponent_raises:
    return None
  shove = opponent_raises[-1]["action"] == "all_in"
  if not shove and not game_state["your_bet_this_street"]:
    return None
  pressure = min(1.0, owed / max(1, game_state["pot"] - owed))
  power = 1.5 + 4 * pressure + min(2, len(opponent_raises) - 1)
  # Keep missed draws and bluffs in the distribution, including on shove lines.
  floor = 0.12 if shove else 0.24
  scores = {
    (first, second): eval7.evaluate(board + [first, second])
    for first, second in combinations(unseen, 2)
  }
  weights = {score: floor + (1 - floor) * p ** power
             for score, p in _hand_percentiles(scores.values())}
  return {pair: weights[score] for pair, score in scores.items()}


def _showdown_probabilities(game_state, opponent=None):
  """Return win and tie probabilities, optionally weighting river hands."""
  hole_cards, board, unseen = _showdown_cards(game_state)
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
  postflop_weights = (
    _postflop_reraise_weights(game_state, board, unseen)
    if opponent and cards_to_come in (1, 2) else None
  )
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
    elif postflop_weights is not None:
      first, second = draw[:2]
      weight = postflop_weights.get((first, second)) or postflop_weights[second, first]
    total_weight += weight
    if our_score > their_score:
      wins += weight
    elif our_score == their_score:
      ties += weight
  return wins / total_weight, ties / total_weight


def _turn_raise_equities(game_state, opponent):
  """Estimate call and reraise equity for small and large turn raises."""
  hole_cards, board, unseen = _showdown_cards(game_state)
  scores = {
    (first, second): eval7.evaluate(board + [first, second])
    for first, second in combinations(unseen, 2)
  }
  ours_by_river = {river: eval7.evaluate(hole_cards + board + [river]) for river in unseen}
  outcomes = {}
  for _ in range(_MONTE_CARLO_TRIALS):
    first, second, river = random.sample(unseen, 3)
    score = scores.get((first, second))
    if score is None:
      score = scores[second, first]
    ours = ours_by_river[river]
    theirs = eval7.evaluate(board + [river, first, second])
    points, trials = outcomes.get(score, (0.0, 0))
    outcomes[score] = (points + (ours > theirs) + 0.5 * (ours == theirs), trials + 1)

  weighted_points = [0.0] * 4
  weighted_trials = [0.0] * 4
  response_weights = opponent.turn_response_weights(game_state, scores.values())
  for score, (points, trials) in outcomes.items():
    for index, weight in enumerate(response_weights[score]):
      weighted_points[index] += points * weight
      weighted_trials[index] += trials * weight
  small_call, small_raise, large_call, large_raise = (
    points / trials for points, trials in zip(weighted_points, weighted_trials)
  )
  return {False: (small_call, small_raise), True: (large_call, large_raise)}


def _river_turn_prior(game_state):
  if game_state["street"] != "river" or not game_state["can_check"]:
    return None
  actions = [event for event in game_state.get("action_log", [])
             if event["street"] == "turn" and event["bot_id"] != game_state["your_bot_id"]]
  if any(event["action"] in ("raise", "all_in") for event in actions):
    return 0.20, 3
  if 6 * sum(event["amount"] for event in actions if event["action"] == "call") >= game_state["pot"]:
    return 0.35, 2
  return None


def _river_raise_equities(game_state, opponent, turn_prior=None):
  """Compute equity against the ranges that continue after a river raise."""
  hole_cards, board, unseen = _showdown_cards(game_state)
  ours = eval7.evaluate(hole_cards + board)
  pairs = list(combinations(unseen, 2))
  scores = [eval7.evaluate(board + list(pair)) for pair in pairs]
  percentiles = dict(_hand_percentiles(scores))
  observed = opponent.river_range_weights(game_state, scores)
  weights = [observed[score] if observed is not None else 1.0 for score in scores]
  if turn_prior is not None:
    floor, power = turn_prior
    turn_scores = [eval7.evaluate(board[:4] + list(pair)) for pair in pairs]
    turn_weights = {score: floor + (1 - floor) * p ** power
                    for score, p in _hand_percentiles(turn_scores)}
    weights = [weight * turn_weights[score] for weight, score in zip(weights, turn_scores)]
  points = [(ours > score) + 0.5 * (ours == score) for score in scores]
  raw_equity = sum(weight * point for weight, point in zip(weights, points)) / sum(weights)
  call_rate = opponent.action_probabilities("river", facing_bet=True)["call"]
  shapes = ((max(0.50, call_rate), 2), (0.12, 5),
            (max(0.24, call_rate), 4), (0.06, 7))
  outcomes = []
  for floor, exponent in shapes:
    continuation = [weight * (floor + (1 - floor) * percentiles[score] ** exponent)
                    for weight, score in zip(weights, scores)]
    outcomes.append(sum(weight * point for weight, point in zip(continuation, points)) / sum(continuation))
  return raw_equity, {False: tuple(outcomes[:2]), True: tuple(outcomes[2:])}


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
  turn_prior = _river_turn_prior(game_state)
  river_equities = None
  if turn_prior is not None:
    equity, river_equities = _river_raise_equities(game_state, _opponent, turn_prior)
  else:
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
    marginal_flop_reraise = (
      game_state["street"] == "flop" and not can_check
      and game_state["your_bet_this_street"] > 0 and equity < 0.60
    )
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
    # Use a tighter continuation range after the opponent has shown aggression.
    continuation_equities = river_equities
    if game_state["street"] == "turn" and not can_check:
      continuation_equities = _turn_raise_equities(game_state, _opponent)
    elif game_state["street"] == "river" and not can_check:
      _, continuation_equities = _river_raise_equities(game_state, _opponent)
    for target in sorted(targets):
      # When already raised on the flop, avoid building a big pot with
      # a marginal hand while the opponent's continuation range is uncertain.
      if marginal_flop_reraise and 5 * (target - current_bet) >= pot_after_call:
        continue
      candidate_responses = responses
      call_equity, reraise_equity = equity, None
      opponent_owed = _opponent_call_chips(game_state, target)
      pot_with_raise = game_state["pot"] + _additional_chips(game_state, target)
      if continuation_equities is not None:
        large_bet = _opponent.is_large_bet(opponent_owed, pot_with_raise)
        call_equity, reraise_equity = continuation_equities[large_bet]
      if game_state["street"] == "river":
        candidate_responses = _opponent.river_response_probabilities(
          opponent_owed, pot_with_raise, responses,
        )
      raise_ev = ev_calc_raise(
        game_state, target, call_equity, response_probabilities=candidate_responses,
        reraise_equity=reraise_equity,
      )
      if raise_ev > best_ev:
        best_action, best_ev = {"action": "raise", "amount": target}, raise_ev
        fold_probability = candidate_responses["fold"]

  best_action["debug"] = {"equity": equity, "fold_probability": fold_probability, "ev": best_ev}
  return best_action
