# OpenSpiel benchmark

This local harness loads the tournament bot's existing `decide(game_state)` and
plays it in OpenSpiel's heads-up no-limit Hold'em engine. OpenSpiel is installed
in a separate Docker image. The tournament bot and its dependencies stay in
their existing locations.

## Build and run from the repository root in PowerShell

```powershell
docker build -t fullhouse-sandbox:latest pokercore/sandbox
docker build -t fullhouse-openspiel-test:local testing/openspiel
docker run --rm --network none --read-only --mount "type=bind,source=$($PWD.Path),target=/kit,readonly" -w /kit fullhouse-openspiel-test:local /usr/local/bin/python3 -B testing/openspiel/benchmark.py --opponent check-call --hands 50 --seed 7
```

Skip the first build if you already have the kit's sandbox image. Use
`--opponent random` or `--opponent fold` for the other reference policies.
`--bot bots/river_sizing_baseline/bot.py` runs a saved version.

The harness plays the same seeded deals from both seats with fixed 25/50 blinds
and fresh stacks each hand. It reports net chips, big blinds per 100 hands, and
the slowest measured decision. It runs the bot in-process for comparison.
Use the kit's validator and sandbox to check tournament compatibility.

The bundled opponents are simple reference policies. Random samples uniformly
from legal actions, which include every integer raise size. To test against a
trained OpenSpiel policy, adapt the opponent construction in `benchmark.py` to
load that policy for the same game and betting configuration.
