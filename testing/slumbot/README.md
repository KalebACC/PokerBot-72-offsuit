# Slumbot benchmark

This benchmark plays the existing tournament entry against the public Slumbot
API. The benchmark process sends the HTTP requests. The entry runs through the
kit's network-disabled Docker sandbox with its normal two-second decision
limit, 768 MB of memory, and half a CPU core.

From the repository root in PowerShell with Docker Desktop running:

```powershell
docker build -t fullhouse-sandbox:latest pokercore/sandbox
py -B testing/slumbot/benchmark.py --hands 12
```

Skip the build if the kit's sandbox image is already available. The benchmark
uses only the Python standard library on the host. It opens one anonymous
session, waits at least half a second between requests, and stops if a request
fails. Slumbot deals the cards and resets both stacks to 200 big blinds each
hand. The script checks that returned winnings agree with the betting history
and reports net chips and big blinds per 100 hands.

Add `--log testing/slumbot/session.jsonl` to record the decisions and payouts.
The log keeps the public betting sequence, pot odds inputs, bot equity and EV,
and the action sent to Slumbot. It records opponent cards only when revealed
at showdown. Use a new filename for each session.

Run this harness from `testing/`. Submit the entry from `my_bots/72_offsuit/`.

[Slumbot API example](https://slumbot.com/sample_api.py)
