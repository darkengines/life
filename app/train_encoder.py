"""Asynchronous training of the world's shared perception encoder.

Runs as its OWN process on the GPU, entirely off the simulation's tick loop:
it reads the experience chunks the simulation has already written, trains,
writes improved encoder weights to a file, and the simulation hot-loads them
whenever they change. Neither side ever waits for the other, which is the
requirement -- the world keeps running at full speed while perception
improves in the background.

WHY THIS OBJECTIVE
Every creature has its own evolved decision layer, but perception is shared,
so the encoder is the one part that can be trained on the whole population's
experience at once. It is trained self-supervised, to predict what happens
next: given the latent of the current senses and the action actually taken,
predict the next sense vector, and predict the reward. No hand-designed
notion of "good behaviour" is involved -- a representation that supports
predicting consequences is a representation that carries the information
control needs, and it cannot bias evolution toward any particular strategy.

This matters because measurement showed evolved brains performing no better
than random ones at most behaviours: each individual was expected to discover
feature extraction from 34 raw channels by mutation alone. Learning the
representation once, from data, is the part that gradient descent is
genuinely better at than evolution.
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
VAR = ROOT / "var"
LOG_DIR = VAR / "experience_log"
WEIGHTS_PATH = VAR / "shared_encoder.npz"
STATUS_PATH = VAR / "trainer_status.json"

LATENT_DIM = 12
TRAIN_INTERVAL = 60.0     # seconds between training rounds
STEPS_PER_ROUND = 400
BATCH = 4096
LR = 1e-3
MIN_ROWS = 4000           # don't bother training on a trickle


class WorldModel(nn.Module):
    """encoder(sense) -> latent, then predict next sense and reward.

    Only the encoder is exported to the simulation; the prediction heads
    exist purely to give the encoder a reason to keep useful information.
    """

    def __init__(self, sense_dim, act_dim, latent_dim):
        super().__init__()
        # The exported encoder is deliberately a single tanh layer, because
        # that is exactly what the engine evaluates per creature per tick.
        # Anything deeper would not survive the trip back.
        self.encoder = nn.Linear(sense_dim, latent_dim)
        self.predict_next = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 128), nn.ReLU(),
            nn.Linear(128, sense_dim),
        )
        self.predict_reward = nn.Sequential(
            nn.Linear(latent_dim + act_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, sense, action):
        z = torch.tanh(self.encoder(sense))
        za = torch.cat([z, action], dim=-1)
        return z, self.predict_next(za), self.predict_reward(za).squeeze(-1)


def load_transitions(max_files=40):
    """Consecutive (state, action, reward, next state) tuples.

    The engine logs each individual for a short run of CONSECUTIVE ticks, so
    rows that share an id and sit one tick apart are a genuine transition.
    Anything else is dropped rather than silently pairing unrelated moments.
    """
    files = sorted(glob.glob(str(LOG_DIR / "*.npz")))[-max_files:]
    S, A, R, S2 = [], [], [], []
    for f in files:
        try:
            d = np.load(f)
        except Exception:
            continue
        ids, ticks = d["ids"], d["ticks"]
        sense, action = d["sense"], d["action"]
        reward = d["reward"] if "reward" in d.files else np.zeros(len(ids), dtype=np.float32)
        order = np.lexsort((ticks, ids))
        ids, ticks = ids[order], ticks[order]
        sense, action, reward = sense[order], action[order], reward[order]
        same = (ids[1:] == ids[:-1]) & (ticks[1:] == ticks[:-1] + 1)
        idx = np.nonzero(same)[0]
        if len(idx):
            S.append(sense[idx]); A.append(action[idx])
            R.append(reward[idx]); S2.append(sense[idx + 1])
    if not S:
        return None
    return (np.concatenate(S), np.concatenate(A),
            np.concatenate(R), np.concatenate(S2))


def write_status(**kw):
    kw["updated"] = time.time()
    try:
        STATUS_PATH.write_text(json.dumps(kw, indent=2))
    except OSError:
        pass


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[trainer] device={dev} "
          f"({torch.cuda.get_device_name(0) if dev == 'cuda' else 'no GPU'})", flush=True)
    model, opt, round_no = None, None, 0

    while True:
        data = load_transitions()
        if data is None or len(data[0]) < MIN_ROWS:
            n = 0 if data is None else len(data[0])
            print(f"[trainer] only {n} transitions available, waiting", flush=True)
            write_status(state="waiting", transitions=n)
            time.sleep(TRAIN_INTERVAL)
            continue

        S, A, R, S2 = data
        sense_dim, act_dim = S.shape[1], A.shape[1]
        if model is None:
            model = WorldModel(sense_dim, act_dim, LATENT_DIM).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            print(f"[trainer] model {sense_dim}->{LATENT_DIM}, act {act_dim}", flush=True)

        S_t = torch.from_numpy(S).float().to(dev)
        A_t = torch.from_numpy(A).float().to(dev)
        R_t = torch.from_numpy(R).float().to(dev)
        S2_t = torch.from_numpy(S2).float().to(dev)

        model.train()
        last = 0.0
        for step in range(STEPS_PER_ROUND):
            i = torch.randint(0, len(S_t), (min(BATCH, len(S_t)),), device=dev)
            _z, pred_next, pred_r = model(S_t[i], A_t[i])
            loss = nn.functional.mse_loss(pred_next, S2_t[i]) \
                + 0.1 * nn.functional.mse_loss(pred_r, R_t[i])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.item())

        # A baseline of "predict no change" makes the loss meaningful: below
        # it, the representation is carrying real predictive information.
        with torch.no_grad():
            naive = float(nn.functional.mse_loss(S_t, S2_t).item())

        w = model.encoder.weight.detach().cpu().numpy().astype(np.float32)
        b = model.encoder.bias.detach().cpu().numpy().astype(np.float32)
        tmp = WEIGHTS_PATH.with_suffix(".tmp.npz")
        np.savez(tmp, w=w.reshape(-1), b=b)
        os.replace(tmp, WEIGHTS_PATH)   # atomic, so the sim never reads a half-written file

        round_no += 1
        print(f"[trainer] round {round_no}: {len(S_t)} transitions, "
              f"loss {last:.5f} vs naive {naive:.5f} -> weights written", flush=True)
        write_status(state="trained", round=round_no, transitions=int(len(S_t)),
                     loss=last, naive_loss=naive, device=dev)
        time.sleep(TRAIN_INTERVAL)


if __name__ == "__main__":
    main()
